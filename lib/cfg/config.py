import os
import datetime
import argparse
from lib.data import getdata

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Dataset
parser.add_argument('--data_root', type=str, default='/mnt/data2/caffe/person_reid/')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=getdata.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=112,
                    help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
parser.add_argument('--train-sample-method', type=str, default="interval", choices=["interval", "random", "evenly"],
                    help="how to sample seq_len frames from a sequence.")
parser.add_argument('--test-sample-method', type=str, default="evenly",
                    choices=["interval", "evenly", "dense", "random"],
                    help="how to sample seq_len frames from a sequence.")

# Optimization options
parser.add_argument('--max-epoch', default=150, type=int,
                    help="maximum epochs to run")
parser.add_argument('--warmup-epoch', default=10, type=int,
                    help="warm-up epochs")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', default=3e-4, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=50, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--resume', type=str, default='', help="checkpoint path")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='tap',
                    help="3d, tap, tws, rnn")
parser.add_argument('--base-model', type=str, default='resnet50')
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])
parser.add_argument('--conv5-stride', type=int, default=1, choices=[1, 2])
parser.add_argument('--no-batch-norm', action='store_true', help="")
parser.add_argument('--pool-type', type=str, default="avg", choices=["avg", "max"])
parser.add_argument('--is-shift', action='store_true', help="")
parser.add_argument('--non-local', action='store_true', help="")
parser.add_argument('--stm', default=[], nargs='+', type=str,
                    help="spatio-temporal-motion module, ['cstm', 'cmm']")

# Miscs
parser.add_argument('--print-freq', type=int, default=100, help="print frequency")
parser.add_argument('--seed', type=int, default=7, help="manual seed")
parser.add_argument('--pretrained-model', type=str, default='', help='need to be set for 3d cnn')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-steps', type=int, default=[100], nargs="+",
                    help="run evaluation for every N epochs (set to -1 to test after training)")
now = datetime.datetime.now()
parser.add_argument('--save-dir', type=str,
                    default=os.path.join('/mnt/data2/caffe/ckpt/videoReID/', '{}'.format(now.strftime("%Y%m%d"))))
parser.add_argument('--ckpt-dir', type=str, default="something")
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')


def get_args():
    args = parser.parse_args()
    return args
