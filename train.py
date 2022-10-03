import os
from pickletools import optimize
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import copy
import random
from utils import get_dataset, get_network, get_eval_pool, get_time, DiffAugment, ParamDiffAug
from reparam_module import ReparamModule
from evaluation import evaluate_bpc
from log import get_logger
from config import setup_hps

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    ##### experiment setup #####
    args = setup_hps(args)
    workdir = os.path.join('results', '%s_BPC_%s_ipc%d'%(args.dataset, args.divergence, args.ipc), 
                            time.strftime('%Y-%m-%d-%I-%M-%S'))
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    logfilename = os.path.join(workdir, time.strftime('train.log'))
    logger = get_logger(logfilename)

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    logger.info("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.distributed = torch.cuda.device_count() > 1


    ##### get dataset #####
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean0, std0, \
        dst_train, dst_test, testloader, loader_train_dict, class_map, \
        class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size

    if args.dsa:
        args.dc_aug_param = None
    args.dsa_param = ParamDiffAug()

    args.batch_syn = num_classes * args.ipc

    logger.info('Hyper-parameters: \n %s' %args.__dict__)
    logger.info('Evaluation model pool: %s' %model_eval_pool)

    ##### organize the real dataset #####
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    logger.info("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        logger.info('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        logger.info('real images channel %d, mean = %.4f, std = %.4f'\
                %(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]
    def get_real_batch(n):
        idx_shuffle = np.random.permutation(len(labels_all))[:n]
        images = images_all[idx_shuffle]
        labels = labels_all[idx_shuffle]
        return images, labels

    ##### initialize pseudocoresets #####
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], \
                    dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_init).to(args.device)

    if args.pix_init == 'real':
        logger.info('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        logger.info('initialize synthetic data from random noise')


    ##### training setup #####
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    # scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=0)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    criterion = nn.CrossEntropyLoss().to(args.device)
    logger.info('%s training begins'%get_time())


    ##### get expert trajectory buffer #####
    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    expert_dir = os.path.join(expert_dir, args.model)
    logger.info("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        logger.info("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)


    ##### training  #####
    for it in range(0, args.Iteration+1):

        ##### Evaluate pseudocoresets #####
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                logger.info('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    logger.info('DSA augmentation strategy: \n %s' %args.dsa_strategy)
                    logger.info('DSA augmentation parameters: \n %s' %args.dsa_param.__dict__)
                else:
                    logger.info('DC augmentation parameters: \n %s' %args.dc_aug_param)

                accs_test = []
                nlls_test = []
                for it_eval in range(args.num_eval):
                    logger.info('%s evaluation iter : %d' %(args.eval_method, it_eval))
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                    args.lr_net = syn_lr.item()
                    loss_test, acc_test= evaluate_bpc(args.eval_method, net_eval, image_syn_eval, label_syn_eval, testloader, args, logger)
                    accs_test.append(acc_test)
                    nlls_test.append(loss_test)
                acc_test_mean = np.mean(np.array(accs_test))
                acc_test_std = np.std(np.array(accs_test))
                nll_mean = np.mean(np.array(nlls_test))
                nll_std = np.std(np.array(nlls_test))

                logger.info('-------------------------\nIter %d Evaluate %d random %s, mean = %.4f(%.4f), nll = %.4f(%.4f)'
                                %(it, len(accs_test), model_eval, acc_test_mean, acc_test_std, nll_mean, nll_std))

            ##### visualize pseudocoresets #####
            with torch.no_grad():
                save_dir = os.path.join(workdir, 'pic')
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                if it == args.Iteration:
                    torch.save(image_syn.cpu(), os.path.join(workdir, "images_%d.pt"%(it)))
                    torch.save(label_syn.cpu(), os.path.join(workdir, "labels_%d.pt"%(it)))

                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std0[ch] + mean0[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, os.path.join(save_dir, 'images_%d.png'%(it)), nrow=args.ipc)


        ##### training #####
        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])


        ##### get expert trajectory #####
        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)


        ##### BPC algorithms #####
        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        y_hat = label_syn.to(args.device)
        indices_chunks = []

        for step in range(args.syn_steps):
            if not indices_chunks:
                indices = torch.randperm(len(image_syn))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()

            x = image_syn[these_indices]
            this_y = y_hat[these_indices]

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            if args.divergence == 'wasserstein':
                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
            else:
                grad = torch.autograd.grad(ce_loss, student_params[-1])[0]

            student_params.append(student_params[-1] - syn_lr * grad)

        if args.divergence == 'wasserstein':   
            param_loss = torch.tensor(0.0).to(args.device)
            param_dist = torch.tensor(0.0).to(args.device)

            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            param_loss /= num_params
            param_dist /= num_params

            grand_loss = param_loss/param_dist

            optimizer_img.zero_grad()
            optimizer_lr.zero_grad()

            grand_loss.backward()

            optimizer_img.step()
            optimizer_lr.step()

            for _ in student_params:
                del _

            if it%10 == 0:
                logger.info('BPC-W %s iter = %04d, loss = %.4f, syn_lr = %.4f' %(get_time(), it, grand_loss.item(), syn_lr.item()))

        elif args.divergence == 'fkl': 
            loss = torch.tensor(0.0).to(args.device)

            for _ in range(args.num_epsilons):
                x = image_syn
                if args.dsa and (not args.no_aug):
                    x = DiffAugment(image_syn, args.dsa_strategy, param=args.dsa_param)
        
                current_param = student_params[-1].clone().detach()  ## unroll gradient detach 
                eps1 = torch.randn_like(student_params[-1]) * args.noise_scale
                ce_loss_student = criterion(student_net(x, flat_param = current_param+eps1), label_syn)            
                
                eps2 = torch.randn_like(student_params[-1]) * args.noise_scale
                ce_loss_teacher = criterion(student_net(x, flat_param = target_params+eps2), label_syn)

                loss += (- ce_loss_student + ce_loss_teacher)/args.num_epsilons
            
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            for _ in student_params:
                del _

            if it%10 == 0:
                logger.info('BPC-fKL %s iter = %04d, loss = %.4f' %(get_time(), it, loss.item()))
        
        elif args.divergence == 'rkl':
            image_syn = image_syn.detach()
            gs = torch.zeros(args.num_epsilons, args.batch_rkl)
            gs_tilde = torch.zeros(args.num_epsilons, args.ipc*num_classes)
            h_tilde = torch.zeros((args.num_epsilons,) + image_syn.size())
            criterion_rkl = nn.CrossEntropyLoss(reduction='none').to(args.device)

            real_images, real_labels = get_real_batch(args.batch_rkl)
            real_images, real_labels = real_images.cuda(), real_labels.cuda() 

            if args.dsa and (not args.no_aug):
                real_images = DiffAugment(real_images, args.dsa_strategy, param=args.dsa_param)
            for num in range(args.num_epsilons):
                current_param = student_params[-1].clone().detach() # unroll gradient detach 
                eps = torch.randn_like(student_params[-1]) * args.noise_scale

                output_real = student_net(
                    real_images, flat_param=current_param + eps)
                gs[num] = - criterion_rkl(output_real, real_labels)

                output_syn = student_net(
                    image_syn, flat_param=current_param + eps)
                gs_tilde[num] = - criterion_rkl(output_syn, label_syn)

                _images = image_syn.clone().data.requires_grad_()
                _images_aug = _images
                if args.dsa and (not args.no_aug):
                    _images_aug = DiffAugment(_images, args.dsa_strategy, param=args.dsa_param)
                outputs = student_net(_images_aug, flat_param=current_param + eps)
                potential = -criterion_rkl(outputs, label_syn)
                grad = torch.autograd.grad(
                    potential, _images, grad_outputs=torch.ones_like(potential))[0]
                h_tilde[num] = grad.clone().detach()

            gs = gs - gs.mean(0)
            gs_tilde = gs_tilde - gs_tilde.mean(0)
            h_tilde = h_tilde - h_tilde.mean(0)

            image_grad = torch.zeros(image_syn.size())
            for s in range(args.num_epsilons):
                image_grad += h_tilde[s] * (gs[s].mean() - gs_tilde[s].mean())
            image_grad /= args.num_epsilons
            image_syn = image_syn + args.lr_img * image_grad.cuda()

            for _ in student_params:
                del _

            if it % 10 == 0:
                logger.info('BPC-rKL %s iter = %04d' % (get_time(), it))
        
        else:
            print('check divergence!')
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=10, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')
    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-5, help='learning rate for updating... learning rate')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_rkl', type=int, default=1000, help='batch size for real data in BPC-rKL')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    #### training hps ####
    parser.add_argument('--divergence', type=str, default='fkl', help='rkl, wasserstein, fkl')
    parser.add_argument('--num_epsilons', type=int, default=30, help='samples for calculating expected potential')
    parser.add_argument('--noise_scale', type=float, default=0.01, help='Gaussian scale')
    parser.add_argument('--lr_init', type=float, default=0.03, help='how to init lr (alpha)')
    parser.add_argument('--expert_epochs', type=int, default=1, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=30, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=20, help='max epoch we can start at')
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    #### test hps ####
    parser.add_argument('--eval_method', type=str, default='hmc', help='hmc, sghmc')
    parser.add_argument('--num_iter', type=int, default=100, help='num iterations')
    parser.add_argument('--num_lf', type=int, default=5, help='leapfrog steps')
    parser.add_argument('--burn', type=int, default=50, help='burnin')
    parser.add_argument('--theta_scale', type=float, default=0.1, help='hmc initial theta scale')
    parser.add_argument('--mom_scale', type=float, default=0.1, help='hmc initial momentum scale')
    parser.add_argument('--eps', type=float, default=1e-2, help='training lr')
    parser.add_argument('--wd', type=float, default=1.5, help='weight decay factor')
    parser.add_argument('--alpha', type=float, default=0.1, help='momentum decay factor')
    parser.add_argument('--T', type=float, default=0.01, help='noise scale')

    args = parser.parse_args()

    main(args)


