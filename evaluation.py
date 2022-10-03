import torch
import torch.nn as nn 
from reparam_module import ReparamModule
import numpy as np
from utils import DiffAugment
import torch.nn.functional as F

def evaluate_bpc(method, net, images_train, labels_train, test_loader, args, logger, testaug=False):
    net = net.to(args.device)
    
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)

    theta = [p.detach().cpu() for p in net.parameters()]
    theta = torch.cat([p.data.to(args.device).reshape(-1) for p in theta], 0)
    theta = torch.randn_like(theta) * args.theta_scale
    theta = theta.requires_grad_(True)

    eps = args.eps
    weight_decay = args.wd
    num_train = labels_train.size()[-1]
    criterion = nn.CrossEntropyLoss().to(args.device)
    samples = []

    ##### sampling #####
    if method == 'hmc':
        num_reject = 0
        net = ReparamModule(net)

        for s in range(args.num_iter):
            
            if testaug :
                images_train = DiffAugment(images_train, args.dsa_strategy, param=args.dsa_param)

            momentum = torch.randn_like(theta) * args.mom_scale
            theta0 = theta
            momentum0 = momentum
            
            K0 = torch.sum(momentum*momentum)/2
            output = net(images_train, flat_param=theta)
            loss = criterion(output, labels_train) 
            U0 = loss + weight_decay*torch.norm(theta)

            for leap_frog_step in range(args.num_lf):
                output = net(images_train, flat_param=theta)
                loss = criterion(output, labels_train)
                U = (loss + weight_decay*torch.norm(theta))
                grads = torch.autograd.grad(U, theta)[0]
                momentum = momentum - 0.5*eps*grads*num_train

                theta = theta + eps*momentum
                theta = theta.detach().data.requires_grad_(True)

                output = net(images_train, flat_param=theta)
                loss = criterion(output, labels_train)
                U = (loss + weight_decay*torch.norm(theta))
                grads = torch.autograd.grad(U, theta)[0]
                momentum = momentum - 0.5*eps*grads*num_train

                acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), labels_train.cpu().data.numpy()))

            K1 = torch.sum(momentum*momentum)/2
            output = net(images_train, flat_param=theta)
            loss = criterion(output, labels_train)
            U1 = loss + weight_decay*torch.norm(theta)

            K_diff = K1-K0
            U_diff = num_train * (-U1+U0)
            if torch.rand(1).cuda() > torch.exp(torch.minimum(torch.tensor(0.0).cuda(), U_diff+K_diff)):
                theta = theta0
                momentum = momentum0
                num_reject += 1 
                print('reject!', K_diff, U_diff)
            if s >= args.burn:
                samples.append(theta)
            if s % (args.num_iter//10) == 0:
                logger.info('%d iter trainng acc: %.2f, training loss: %.4f'%(s, acc/num_train, loss.item()))
        logger.info('accept: %.4f'%(1-num_reject/args.num_iter))
    
    elif method == 'sghmc':
        net = ReparamModule(net)        
        momentum = torch.randn_like(theta) * args.mom_scale
        for s in range(args.num_iter):

            for leap_frog_step in range(args.num_lf):
                theta = theta + eps*momentum
                theta = theta.detach().data.requires_grad_(True)

                images_train_aug = DiffAugment(images_train, args.dsa_strategy, param=args.dsa_param)

                output = net(images_train_aug, flat_param=theta)
                loss = criterion(output, labels_train)
                U = (loss + weight_decay*torch.norm(theta))
                grads = torch.autograd.grad(U, theta)[0]
                noise = torch.randn_like(grads) * np.sqrt(2*args.alpha*args.T)
                momentum = (1-args.alpha)*momentum - eps*grads*num_train + noise

                acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), labels_train.cpu().data.numpy()))

            if s >= args.burn :
                samples.append(theta)
            if s % (args.num_iter//10) == 0:
                logger.info('%d iter trainng acc: %.2f, training loss: %.4f'%(s, acc/(labels_train.size()[-1]), loss.item()))
    else:
        print('check evaluation method!')


    ##### test #####
    img_net_classes = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    if args.dataset == "ImageNet":
        class_map = {x: i for i, x in enumerate(img_net_classes)}

    with torch.no_grad():
        loss_avg, acc_avg, num_exp, avg_brier = 0, 0, 0, 0

        bins0 = np.linspace(start=0, stop=1.0, num=30)
        o = 0
        conf = 0
        for i_batch, datum in enumerate(test_loader):
            img = datum[0].float().to(args.device)
            lab = datum[1].long().to(args.device)
            if args.dataset == "ImageNet":
                lab = torch.tensor([class_map[x.item()] for x in lab]).to(args.device)
            n_b = lab.shape[0]

            logitss = []
            nlls = []
            for theta in samples:
                output = net(img, flat_param=theta)
                logitss.append(output)
                nll = criterion(output, lab)
                nlls.append(nll)
            
            logitss = torch.stack(logitss)
            nlls = torch.stack(nlls)
            
            ll = torch.logsumexp(-nlls, 0).cpu().data.numpy() - np.log(len(samples))
            loss = -ll
            output = F.softmax(logitss, dim=-1).mean(0).cpu().numpy()
            acc = np.sum(np.equal(np.argmax(output, axis=-1), lab.cpu().data.numpy()))

            lab_onehot = F.one_hot(lab, num_classes=10)
            brier = ((F.softmax(logitss, dim=-1).mean(0) - lab_onehot)**2).sum(1).mean()

            conf += (np.amax(output,  axis=-1)).sum()
            pred_y = np.argmax(output, axis=-1)
            correct = (pred_y == lab.cpu().data.numpy()).astype(np.float32)

            prob_y = np.max(output, axis=-1)
            bins = np.digitize(prob_y, bins=bins0, right=True)
            for b in range(30):
                mask = bins == b
                if np.any(mask):
                    o += np.abs(np.sum(correct[mask] - prob_y[mask]))

            loss_avg += loss*n_b
            acc_avg += acc
            num_exp += n_b
            avg_brier += brier.item() * n_b

        loss_avg /= num_exp
        acc_avg /= num_exp
        o /= num_exp
        avg_brier /= num_exp
        logger.info('%d samples %s test acc: %.4f, nll: %.4f, ece: %.4f, brier: %.4f'%(len(samples), method, acc_avg, loss_avg, o, avg_brier))

    return loss_avg, acc_avg