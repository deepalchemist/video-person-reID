from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import random
from numpy.random import randint


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['interval', 'evenly', 'random', 'dense']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num_frame = len(img_paths)
        if self.sample == "interval":
            """
            Divide input videos into T segments of equal durations,
            then randomly sample one frame from each segment to obtain 
            the input sequence with T frames.
            """
            indices = []
            if num_frame < self.seq_len:
                indices = list(range(num_frame))
                indices.extend([indices[-1]] * (self.seq_len - num_frame))
            else:
                segments = np.array_split(range(num_frame), self.seq_len)
                for seg in segments:
                    indices.append(np.random.choice(seg))
            assert len(indices) == self.seq_len
            # read images
            imgs = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)  # (seq_len 3 h w)

            return imgs, pid, camid

        elif self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = list(range(num_frame))
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)

            # read images
            imgs = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)  # (seq_len 3 h w)

            return imgs, pid, camid

        elif self.sample == 'evenly':
            # Evenly samples seq_len images from a tracklet
            if num_frame >= self.seq_len:
                num_frame -= num_frame % self.seq_len
                indices = np.arange(0, num_frame, num_frame / self.seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_frame)
                num_pads = self.seq_len - num_frame
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_frame - 1)
                    ]
                )
            assert len(indices) == self.seq_len

            # read images
            imgs = []
            for index in indices:
                img_path = img_paths[int(index)]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)  # img must be torch.Tensor
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)

            return imgs, pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, 
            batch_size needs to be set to 1. This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = list(range(num_frame))
            indices_list = []
            while num_frame - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)

            imgs_list = []
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                # imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))
