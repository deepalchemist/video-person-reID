from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from lib.dataset import data_manager, transforms as T
from video_loader import VideoDataset
from lib.models.model_manager import init_model
from lib.models import resnet3d
from losses import CrossEntropyLabelSmooth, TripletLoss
from eval_metrics import evaluate
from lib.dataset.samplers import RandomIdentitySampler
import lib.utils as util
from lib.dataset import init_dataset

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Dataset
parser.add_argument('--data_root', type=str, default='/mnt/data2/caffe/person_reid/')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=112,
                    help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
parser.add_argument('--train-sample-method', type=str, default="interval", choices=["interval", "random", "evenly"],
                    help="how to sample seq_len frames from a sequence.")
parser.add_argument('--test-sample-method', type=str, default="evenly", choices=["interval", "evenly", "dense"],
                    help="how to sample seq_len frames from a sequence.")
# Optimization options
parser.add_argument('--max-epoch', default=800, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=200, type=int,
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
parser.add_argument('-a', '--arch', type=str, default='resnet50tp',
                    help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])
parser.add_argument('--conv5-stride', type=int, default=2, choices=[1, 2])
parser.add_argument('--batch-norm', action='store_true', help="")
parser.add_argument('--pool-type', type=str, default="avg", choices=["avg", "max"])
parser.add_argument('--is-shift', action='store_true', help="")
parser.add_argument('--non-local', action='store_true', help="")
parser.add_argument('--stm', default=[], nargs='+', type=str,
                    help="spatio-temporal-motion module, ['cstm', 'cmm']")

# Miscs
parser.add_argument('--print-freq', type=int, default=80, help="print frequency")
parser.add_argument('--seed', type=int, default=7, help="manual seed")
parser.add_argument('--pretrained-model', type=str, default='', help='need to be set for resnet3d models')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=50,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
now = datetime.datetime.now()
parser.add_argument('--save-dir', type=str, default=os.path.join('./ckpt', '{}'.format(now.strftime("%Y-%m-%d"))))
parser.add_argument('--ckpt-dir', type=str, default="something")
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


def main():
    util.init_random_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    args.ckpt_dir = "test" if args.evaluate else args.ckpt_dir
    args.save_dir = osp.join(args.save_dir, args.ckpt_dir)
    util.mkdir_if_missing(args.save_dir)

    if not args.evaluate:
        sys.stdout = util.Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = util.Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    # =========================================================================================
    print("* Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # train_set = init_dataset(args.dataset, root=args.data_root, seq_len=args.seq_len,
    #                          sample_method='random', mode='train', transform=transform_train)
    # query_set = init_dataset(args.dataset, root=args.data_root, seq_len=args.seq_len,
    #                          sample_method='evenly', mode='query', transform=transform_test, verbose=False)
    # gallery_set = init_dataset(args.dataset, root=args.data_root, seq_len=args.seq_len,
    #                            sample_method='evenly', mode='gallery', transform=transform_test, verbose=False)

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        # train_set,
        VideoDataset(dataset.train, seq_len=args.seq_len, sample=args.train_sample_method, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, batch_size=args.train_batch, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        # query_set,
        VideoDataset(dataset.query, seq_len=args.seq_len, sample=args.test_sample_method, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        # gallery_set,
        VideoDataset(dataset.gallery, seq_len=args.seq_len, sample=args.test_sample_method, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False,
    )

    # =========================================================================================
    # Initialize model, optimizer, and scheduler
    print("* Initializing model: {}".format(args.arch))
    if args.arch == 'resnet503d':
        model = resnet3d.resnet50(num_classes=dataset.num_train_pids, sample_width=args.width,
                                  sample_height=args.height, sample_duration=args.seq_len)
        if not os.path.exists(args.pretrained_model):
            raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
        print("Loading checkpoint from '{}'".format(args.pretrained_model))
        checkpoint = torch.load(args.pretrained_model)
        state_dict = {}
        for key in checkpoint['state_dict']:
            if 'fc' in key: continue
            state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict, strict=False)
    elif args.arch == 'resnet50ts':
        model = init_model(name=args.arch,
                           num_classes=dataset.num_train_pids, num_segments=args.seq_len, modality="RGB",
                           base_model="resnet50", conv5_stride=args.conv5_stride, bn=args.batch_norm,
                           pool_type=args.pool_type,
                           consensus_type="avg", loss={'xent', 'htri'},
                           non_local=args.non_local, stm=args.stm,
                           is_shift=args.is_shift, shift_div=8, shift_place="blockres",
                           fc_lr5=True,
                           )
    else:
        model = init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=args.margin)

    # param_groups = model.get_optim_policies(args.lr, args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    assert args.stepsize > 0
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma, last_epoch=-1)

    # =========================================================================================
    # optionally resume from a checkpoint
    best_rank1 = -np.inf
    start_epoch = args.start_epoch
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_rank1 = checkpoint['rank1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma,
                                        last_epoch=checkpoint['epoch'])
        print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        del checkpoint

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("* Evaluating")
        with torch.no_grad():
            test(model, queryloader, galleryloader, args.pool, use_gpu)
        return

    if args.arch == 'resnet503d':
        torch.backends.cudnn.benchmark = False

    # =========================================================================================
    print("\n* Start training")
    start_time = time.time()
    for epoch in range(start_epoch, args.max_epoch):
        epoch_start_time = time.time()
        train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)

        lr_used = [pg['lr'] for pg in optimizer.param_groups]
        lr_msg = 'used lr: '
        for item in lr_used: lr_msg += '%.0E ' % (item)
        print('* end of epoch {}/{}, time taken: {:.0f} sec, {}'.format(
            epoch, args.max_epoch, time.time() - epoch_start_time, lr_msg))

        scheduler.step(epoch + 1)  # setting lr for next epoch, self.last_epoch==epoch+1

        if args.eval_step > 0 and epoch % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            print("* test")
            with torch.no_grad():
                rank1 = test(model, queryloader, galleryloader, args.pool, use_gpu)
            is_best = rank1 > best_rank1
            if is_best: best_rank1 = rank1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            util.save_checkpoint({
                'rank1': rank1,
                'epoch': epoch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
            }, is_best, osp.join(args.save_dir, 'latest_model.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Checkpoints are saved to {}".format(args.save_dir))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    model.train()
    losses = util.AverageMeter()

    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()  # (N T C H W)
        outputs, features = model(imgs)
        if args.htri_only:
            # only use hard triplet loss to train the network
            loss = criterion_htri(features, pids)
        else:
            # combine hard triplet loss with cross entropy loss
            xent_loss = criterion_xent(outputs, pids)
            htri_loss = criterion_htri(features, pids)
            loss = xent_loss + htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print("batch {}/{} loss {:.3f}({:.3f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))


def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()

    qf, q_pids, q_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(tqdm(queryloader)):
        if use_gpu:
            imgs = imgs.cuda()
        # b=1, n=number of clips, s=16
        b, s, c, h, w = imgs.size()

        # b, n, s, c, h, w = imgs.size()
        # assert (b == 1)
        # imgs = imgs.view(b * n, s, c, h, w)

        features = model(imgs)
        features = features.view(b, -1)  # TODO(note) n to b
        # features = torch.mean(features, dim=0).unsqueeze(0)
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
    qf = torch.cat(qf, dim=0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(tqdm(galleryloader)):
        if use_gpu:
            imgs = imgs.cuda()
        b, s, c, h, w = imgs.size()

        # b, n, s, c, h, w = imgs.size()
        # imgs = imgs.view(b * n, s, c, h, w)
        # assert (b == 1)

        features = model(imgs)
        features = features.view(b, -1)  # TODO(note) n to b
        # if pool == 'avg':
        #     features = torch.mean(features, dim=0).unsqueeze(0)
        # else:
        #     features, _ = torch.max(features, dim=0).unsqueeze(0)
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.cat(gf, dim=0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    return cmc[0]


if __name__ == '__main__':
    main()
