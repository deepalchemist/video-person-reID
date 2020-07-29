import os
import cv2
import sys
import math
import os.path as osp
import time
import errno
import json
from collections import OrderedDict
import warnings
import random
import numpy as np
import shutil
from PIL import Image

import torch

"""#############
### Training ###
#############"""


def init_random_seed(seed):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # lead to torch.initial_seed==opt.seed, pytorch CPU
    torch.cuda.manual_seed(seed)  # pytorch GPU
    torch.cuda.manual_seed_all(seed)  # pytorch GPU
    torch.backends.cudnn.benchmark = False  # accelerate computing
    torch.backends.cudnn.deterministic = True  # avoid inference performance variation
    torch.backends.cudnn.enabled = True

"""##################
### Visualization ###
##################"""


def add_border(im, border_width, value):
    """Add color border around an image. The resulting image size is not changed.
    Args:
      im: numpy array with shape [3, im_h, im_w]
      border_width: scalar, measured in pixel
      value: scalar, or numpy array with shape [3]; the color of the border
    Returns:
      im: numpy array with shape [3, im_h, im_w]
    """
    assert (im.ndim == 3) and (im.shape[0] == 3)
    im = np.copy(im)

    if isinstance(value, np.ndarray):
        # reshape to [3, 1, 1]
        value = value.flatten()[:, np.newaxis, np.newaxis]
    im[:, :border_width, :] = value
    im[:, -border_width:, :] = value
    im[:, :, :border_width] = value
    im[:, :, -border_width:] = value

    return im


def save_vid_rank_result(query, top_gallery, save_path):
    """Save a query and its rank list as an image.
    Args:
        query (1D array): query sequence paths
        top_gallery (2D array): top gallery sequence paths
        save_path:
    """
    assert len(query) % 2 == 0
    n_cols = len(query) // 2

    query_id = int(query[0].split('/')[-2])
    top10_ids = [int(p[0].split('/')[-2]) for p in top_gallery]

    q_images = [read_im(q) for q in query]
    q_im = make_img_grid(q_images, space=4, n_cols=n_cols, pad_val=255)
    images = [q_im]
    for gallery_path, gallery_id in zip(top_gallery, top10_ids):
        g_images = [read_im(g) for g in gallery_path]
        g_im = make_img_grid(g_images, space=4, n_cols=n_cols, pad_val=255)

        # Add green boundary to true positive, red to false positive
        color = np.array([0, 255, 0]) if query_id == gallery_id else np.array([255, 0, 0])
        g_im = add_border(g_im, 3, color)
        images.append(g_im)

    im = make_QGimg_list(images, space=4, pad_val=255)
    im = im.transpose(1, 2, 0)
    Image.fromarray(im).save(save_path)


def make_img_grid(ims, space, n_cols, pad_val):
    """Make a grid of images with space in between.
    Args:
      ims: a list of [3, im_h, im_w] images
      space: the num of pixels between two images
      pad_val: scalar, or numpy array with shape [3]; the color of the space
    Returns:
      ret_im: a numpy array with shape [3, H, W]
    """
    n_rows = math.ceil(len(ims) / n_cols)
    assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
    h, w = ims[0].shape[1:]
    H = h * n_rows + space * (n_rows - 1)
    W = w * n_cols + space * (n_cols - 1)
    if isinstance(pad_val, np.ndarray):
        # reshape to [3, 1, 1]
        pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]
    ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)

    for idx in range(len(ims)):
        curr_row = idx // n_cols
        curr_col = idx % n_cols
        start_h = curr_row * (h + space)
        start_w = curr_col * (w + space)
        ret_im[:, start_h:(start_h + h), start_w:(start_w + w)] = ims[idx]
    return ret_im


def make_QGimg_list(ims, space, pad_val):
    """Make a grid of images with space in between.
    Args:
      ims: a list of [3, im_h, im_w] images
      space: the num of pixels between two images
      pad_val: scalar, or numpy array with shape [3]; the color of the space
    Returns:
      ret_im: a numpy array with shape [3, H, W]
    """
    n_cols = len(ims)
    k_space = 5  # k_space means q_g space
    assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
    h, w = ims[0].shape[1:]
    H = h
    W = w * n_cols + space * (n_cols - 2) + k_space * space
    if isinstance(pad_val, np.ndarray):
        # reshape to [3, 1, 1]
        pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]
    ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)

    ret_im[:, 0:h, 0:w] = ims[0]  # query image

    start_w = w + k_space * space
    for im in ims[1:]:
        end_w = start_w + w
        ret_im[:, 0:h, start_w:end_w] = im
        start_w = end_w + space
    return ret_im


def read_im(im_path):
    # shape [H, W, 3]
    im = np.asarray(Image.open(im_path))
    # Resize to (im_h, im_w) = (128, 64)
    resize_h_w = (128, 64)
    if (im.shape[0], im.shape[1]) != resize_h_w:
        im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
    # shape [3, H, W]
    im = im.transpose(2, 0, 1)
    return im


"""#########
### Misc ###
#########"""


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(path))
            pass
    return img


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
