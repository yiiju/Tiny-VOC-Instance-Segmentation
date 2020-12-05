import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument("--gpuid", type=int, nargs='+', default=8,
                    help="ids of gpus to use")

# optimizer setting
parser.add_argument('--eps', type=float, default=1e-8,
                    help='The eps in Adam optimizer')
parser.add_argument('--lr_rate', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--decay', type=float, default=999999,
                    help='Learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='Learning rate decay factor for step decay')

# trining setting
parser.add_argument('--save_every', type=int, default=1,
                    help='Save period')
parser.add_argument("--epochs", type=int, default=20,
                    help="number of epochs")
parser.add_argument("--batch_size", type=int, default=4,
                    help="size of each image batch")
parser.add_argument("--print_freq", type=int, default=100,
                    help="frequence of print batch metric")

# log setting
parser.add_argument('--save_dir', type=str, default='save_dir',
                    help='Directory to save log, arguments, models and images')
parser.add_argument('--reset', type=str2bool, default=False,
                    help='Delete save_dir to create a new one')
parser.add_argument('--log_file_name', type=str, default='train.log',
                    help='Log file name')
parser.add_argument('--logger_name', type=str, default='maskrcnn',
                    help='Logger name')

# test / finetune setting
parser.add_argument('--test', type=str2bool, default=False,
                    help='Test mode')
parser.add_argument("--checkpoint", type=str,
                    default="checkpoints/maskrcnn/20.pth",
                    help="load checkpoint model")
parser.add_argument("--outjson", type=str, default="new.json",
                    help="path to store the result")

args = parser.parse_args()
