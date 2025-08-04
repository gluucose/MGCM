import os
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    # ******************************************  Basic parameters  ********************************************
    parser.add_argument('--imgs_pt_root', default='../imgs_pt/luad', help="imgs_pt are saved here: luad | blca| lusc")
    parser.add_argument('--datasets_csv_root', default='../datasets_csv/luad', help="datasets_csv are saved here: luad | blca| lusc")
    parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints/luad', help='models are saved here: luad | blca| lusc')
    parser.add_argument('--exp_name', type=str, default='surv', help='project: Multimodal for survival prediction')
    parser.add_argument('--model_name', type=str, default='mgcm', help="mode: mgcm | lowomic | midomic | highomic")
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    # *****************************************   Initialization  ***********************************************
    parser.add_argument('--init_type', type=str, default='none', help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--reg_type', default='all', type=str, help="regularization type")
    parser.add_argument('--act_type', type=str, default='Sigmoid', help='activation function')
    parser.add_argument('--act_type_gene', type=str, default="none", help='activation function of Gene Network')
    parser.add_argument('--measure', default=1, type=int, help='disables measure while training (make program faster)')
    parser.add_argument('--save_at', type=int, default=5, help="Save model checkpoints every N epochs.")
    parser.add_argument('--verbose', default=1, type=int)
    
    # ***********************************  Optimizer & Learning Rate  *********************************************
    parser.add_argument('--optimizer_type', type=str, default='adam', help='Optimizer type, typically "adam".')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1  for Adam optimizer. 0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2  for Adam optimizer. 0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Used only when optimizer is Adam. L2 Regularization on weights.')
    parser.add_argument('--lr_policy', default='linear', type=str, help='Learning rate policy: linear, step, cosine')
    parser.add_argument('--lr', default=1.2e-3, type=float, help='Initial learning rate.  [1e-3 ~ 2e-3] for Adam | 1e-3 for AdaBound')
    parser.add_argument('--final_lr', default=0.1, type=float, help='Used only when optimizer is AdaBound')
    
    # ***********************************  Training Process  *********************************************
    parser.add_argument('--finetune', default=1, type=int, help='1: enable, 0: disable')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training/testing.')
    parser.add_argument('--epoch_count', type=int, default=1, help='Start epoch number for resuming training.')
    parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=30, help='# of iter to linearly decay learning rate to zero.')
    
    # ************************************  Loss Function  **************************************
    parser.add_argument('--lambda_surv', type=float, default=1)
    parser.add_argument('--lambda_cox', type=float, default=1)
    parser.add_argument('--lambda_reg', type=float, default=3e-4)
    
    # *********************************************   Network  **************************************************
    parser.add_argument('--alpha', default=0.2, type=float, help='Used in the leaky relu')
    parser.add_argument('--omic_dim', type=int, default=1, help="Input size for omic vector")
    parser.add_argument('--dropout_gene', default=0.2, type=float, help='Dropout rate of Gene Network')
    parser.add_argument('--path_in', type=int, default=512, help="Input the dims of the Path Layer: 512")
    parser.add_argument('--path_out', type=int, default=256, help="Output the dims of the Path Layer: 512 |256 | 128")
    parser.add_argument('--fusion_dim', type=int, default=256, help='Dims of multimodal fusion feature: 512 | 256 | 128')
    parser.add_argument('--lin_in', type=int, default=256, help="Input the dims of Linear Layer: 512 | 256 | 128")
    parser.add_argument('--dim_fc', type=int, default=128, help="Dims of Fully Connected Layer：256 | 128 | 64")
    parser.add_argument('--lin_out', type=int, default=64, help="Input the dims of Linear Layer：128 | 64 | 32")
    parser.add_argument('--dropout_fc', default=0.25, type=float, help='Dropout rate of FC Layer')

    opt = parser.parse_known_args()[0]
    print_options(parser, opt)
    opt = parse_gpuids(opt)
    return opt


def print_options(parser, opt):
    message = ''
    message += '----------------- Options ---------------\n'

    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)

        if v != default:
            comment = '\t[default: %s]' % str(default)

        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    expr_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)
    mkdirs(expr_dir)

    file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def parse_gpuids(opt):
    # set gpu ids
    str_ids = opt.gpu_ids.splits(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
        
    return opt


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
