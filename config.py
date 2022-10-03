def setup_hps(args):
    ### CIFAR10 ###
    if args.ipc == 1:
        args.max_start_epoch = 2
        args.syn_steps = 50
        args.lr_init = 0.01
        args.lr_lr = 1e-7
        if args.divergence == 'wasserstein':
            args.expert_epochs = 2
        elif args.divergence == 'rkl':
            args.num_epsilons = 10
        args.num_iter = 20
        args.num_lf = 20
        args.burn = 10
        args.mom_scale = 0.01
        args.eps = 0.05
        args.wd = 0.5
        if args.eval_method == 'sghmc':
            args.num_lf = 5
            args.mom_scale = 0.1
            args.eps = 0.03
            args.wd = 1.0
    
    elif args.ipc == 20:
        args.max_start_epoch = 30
        if args.divergence == 'wasserstein':
            args.expert_epochs = 2
        elif args.divergence == 'rkl':
            args.num_epsilons = 10
        if args.eval_method == 'sghmc':
            args.wd = 1.0
        
    return args
