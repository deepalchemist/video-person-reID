from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import numpy as np
from tqdm import tqdm
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from lib.data import getdata, transforms as T
from lib.data.loader import VideoDataset
from lib.loss.losses import CrossEntropyLabelSmooth, TripletLoss
from lib.utils.metrics import evaluate
from lib.data.samplers import RandomIdentitySampler
import lib.utils.utils as util
from lib.cfg.config import get_args
from lib.model.getmodel import get_model
from lib.utils.scheduler import WarmUpLR, update_lr


def main(cfg):
    util.init_random_seed(cfg.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_devices
    use_gpu = torch.cuda.is_available()
    if cfg.use_cpu:
        use_gpu = False

    cfg.ckpt_dir = "test" if cfg.evaluate else cfg.ckpt_dir
    cfg.save_dir = osp.join(cfg.save_dir, cfg.ckpt_dir)
    util.mkdir_if_missing(cfg.save_dir)

    if not cfg.evaluate:
        sys.stdout = util.Logger(osp.join(cfg.save_dir, 'log_train.txt'))
    else:
        sys.stdout = util.Logger(osp.join(cfg.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(cfg))

    if use_gpu:
        print("Currently using GPU {}".format(cfg.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    # --------------------------------------------------------------------------------------------
    print("* Initializing dataset {}".format(cfg.dataset))
    dataset = getdata.init_dataset(name=cfg.dataset)
    cfg.num_train_pids = dataset.num_train_pids

    transform_train = T.Compose([
        T.Random2DTranslation(cfg.height, cfg.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((cfg.height, cfg.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        # train_set,
        VideoDataset(dataset.train, seq_len=cfg.seq_len, sample=cfg.train_sample_method, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, batch_size=cfg.train_batch, num_instances=cfg.num_instances),
        batch_size=cfg.train_batch, num_workers=cfg.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        # query_set,
        VideoDataset(dataset.query, seq_len=cfg.seq_len, sample=cfg.test_sample_method, transform=transform_test),
        batch_size=cfg.test_batch, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        # gallery_set,
        VideoDataset(dataset.gallery, seq_len=cfg.seq_len, sample=cfg.test_sample_method, transform=transform_test),
        batch_size=cfg.test_batch, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False,
    )

    # --------------------------------------------------------------------------------------------
    # Initialize model, optimizer, and scheduler
    print("* Initializing model: {}".format(cfg.arch))
    model = get_model(cfg)
    print("Model size: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=cfg.margin)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    assert cfg.stepsize > 0
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.stepsize, gamma=cfg.gamma, last_epoch=-1)
    if cfg.warmup_epoch > 0:
        scheduler = WarmUpLR(optimizer, scheduler, cfg.warmup_epoch, len(trainloader))

    # --------------------------------------------------------------------------------------------
    # optionally resume from a checkpoint
    best_rank1 = -np.inf
    start_epoch = cfg.start_epoch
    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_rank1 = checkpoint['rank1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.stepsize, gamma=cfg.gamma,
                                        last_epoch=checkpoint['epoch'])
        print("loaded checkpoint '{}' (epoch {})".format(cfg.resume, checkpoint['epoch']))
        del checkpoint

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if cfg.evaluate:
        print("* Evaluating")
        with torch.no_grad():
            evaluate(model, queryloader, galleryloader, cfg.pool, use_gpu)
        return

    if cfg.arch == '3d':
        torch.backends.cudnn.benchmark = False

    # --------------------------------------------------------------------------------------------
    print("\n* Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.max_epoch):
        epoch_start_time = time.time()
        update_lr(scheduler, epoch, n_iter=None)
        train_one_epoch(
            model,
            epoch,
            optimizer,
            scheduler,
            trainloader,
            cfg.warmup_epoch,
            criterion_xent,
            criterion_htri,
            use_gpu
        )

        lr_msg = 'used lr: '
        for item in [pg['lr'] for pg in optimizer.param_groups]:
            lr_msg += '%.0E ' % (item)
        print('* end of epoch {}/{}, time taken: {:.0f} sec, {}'.format(
            epoch, cfg.max_epoch, time.time() - epoch_start_time, lr_msg))

        # scheduler.step(epoch + 1)  # setting lr for next epoch, self.last_epoch==epoch+1

        if epoch in cfg.eval_steps or (epoch + 1) == cfg.max_epoch:
            print("* evaluate")
            with torch.no_grad():
                rank1 = eval(model, queryloader, galleryloader, use_gpu)
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
            }, is_best, osp.join(cfg.save_dir, 'latest.pth'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Checkpoints are saved to {}".format(cfg.save_dir))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train_one_epoch(model,
                    epoch,
                    optimizer,
                    scheduler,
                    trainloader,
                    warm_up,
                    criterion_xent,
                    criterion_htri,
                    use_gpu=True):
    model.train()
    losses = util.AverageMeter()
    n_batch = len(trainloader)
    epoch_iter = 0
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        epoch_iter += 1
        total_iter = epoch * n_batch + epoch_iter
        if warm_up > 0 and epoch < warm_up:
            update_lr(scheduler, epoch, total_iter)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()  # (N T C H W)

        outputs, features = model(imgs)

        # combine hard triplet loss with cross entropy loss
        xent_loss = criterion_xent(outputs, pids)
        htri_loss = criterion_htri(features, pids)
        loss = xent_loss + htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), pids.size(0))

        if total_iter % 100 == 0:
            print("batch {}/{} loss {:.3f}({:.3f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))


@torch.no_grad()
def eval(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()

    qf, q_pids, q_camids, q_paths = [], [], [], []
    for batch_idx, (imgs, pids, camids, img_paths) in enumerate(tqdm(queryloader)):
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
        q_paths.append(np.asarray(img_paths).transpose())

    qf = torch.cat(qf, dim=0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    q_paths = np.concatenate(q_paths)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids, g_paths = [], [], [], []
    for batch_idx, (imgs, pids, camids, img_paths) in enumerate(tqdm(galleryloader)):
        if use_gpu:
            imgs = imgs.cuda()
        b, s, c, h, w = imgs.size()

        # b, n, s, c, h, w = imgs.size()
        # imgs = imgs.view(b * n, s, c, h, w)
        # assert (b == 1)

        features = model(imgs)
        features = features.view(b, -1)
        # if pool == 'avg':
        #     features = torch.mean(features, dim=0).unsqueeze(0)
        # else:
        #     features, _ = torch.max(features, dim=0).unsqueeze(0)
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
        g_paths.append(np.asarray(img_paths).transpose())

    gf = torch.cat(gf, dim=0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    g_paths = np.concatenate(g_paths)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, q_paths, g_paths, plot_ranking=True)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    return cmc[0]


if __name__ == '__main__':
    args = get_args()
    main(args)
