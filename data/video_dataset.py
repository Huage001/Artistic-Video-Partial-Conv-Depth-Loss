import os.path
import numpy as np
import random
import torch
import struct
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_video_dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_and_resize_flow(flow_path, flow_size):

    flo = open(flow_path, 'rb')
    head = flo.read(4)
    assert head == b'PIEH'
    width = int.from_bytes(flo.read(4), 'little')
    height = int.from_bytes(flo.read(4), 'little')
    flo_u = np.zeros((height, width))
    flo_v = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            flo_u[i][j] = struct.unpack('<f', flo.read(4))[0]
            flo_v[i][j] = struct.unpack('<f', flo.read(4))[0]
    flo.close()
    flo_u = flo_u.reshape((1, height, width, 1)).transpose((0, 3, 1, 2)) / (width / 2)
    flo_v = flo_v.reshape((1, height, width, 1)).transpose((0, 3, 1, 2)) / (height / 2)
    flo = torch.from_numpy(np.concatenate((flo_u, flo_v), axis=1))
    flo = F.interpolate(flo, size=flow_size, mode='bicubic', align_corners=False)
    flo = flo[0].numpy().transpose((1, 2, 0))
    grid = np.meshgrid(np.linspace(-1, 1, flow_size[1]), np.linspace(-1, 1, flow_size[0]))
    flo[:, :, 0] += grid[0]
    flo[:, :, 1] += grid[1]
    flo = torch.from_numpy(flo).type(torch.float)
    return flo


class VideoDataset(BaseDataset):
    """
    This dataset class can load video frame dataset.

    There should be three sub-folders under the dataroot, one for original frames, one for optical flow, one for
    mask, and one for style. Things belong to one video should be at one one sub-folders.

    e.g. dataroot/frame/video1/frame1, dataroot/flow/video1/frame1, ...
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.dir_frame = os.path.join(opt.dataroot, 'frame')
        self.dir_flow = os.path.join(opt.dataroot, 'flow')
        self.dir_mask = os.path.join(opt.dataroot, 'mask')
        self.dir_style = os.path.join(opt.dataroot, 'style')
        self.dir_last_fake = os.path.join(opt.dataroot, 'last_fake' + opt.suffix)
        self.frames, self.flows, self.masks, self.styles, self.lasts_fake = make_video_dataset(
            self.dir_frame, self.dir_flow, self.dir_mask, self.dir_style, self.dir_last_fake)
        assert opt.output_nc == 3 and opt.input_nc == 3, 'Numbers of input and output channels must be 3'
        self.transform = get_transform(self.opt)
        self.lengths = [len(video) for video in self.frames]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains frame, flow, and mask
        If this is the first frame of a video, flow will be None.
        """
        video_id = 0
        frame_id = 0
        temp = 0
        is_last = False
        for i in range(len(self.lengths)):
            if index < temp + self.lengths[i]:
                if index == temp + self.lengths[i] - 1:
                    is_last = True
                video_id = i
                frame_id = index - temp
                break
            temp += len(self.frames[i])
        frame = Image.open(self.frames[video_id][frame_id]).convert('RGB')
        frame = self.transform(frame)
        style_id = random.randint(0, len(self.styles) - 1)
        style = Image.open(self.styles[style_id]).convert('RGB')
        style = self.transform(style)
        if frame_id == 0:
            mask = 0
            flow = 0
            last_fake = self.transform(Image.open(self.lasts_fake[video_id]).convert('RGB'))
        else:
            flow = self.flows[video_id][frame_id - 1]
            flow = read_and_resize_flow(flow, (self.opt.crop_size[0], self.opt.crop_size[1]))
            mask = self.masks[video_id][frame_id - 1]
            mask = np.array(Image.open(mask))
            height = mask.shape[0]
            width = mask.shape[1]
            if self.opt.adv_mask:
                mask = 1 - mask.reshape((1, 1, height, width)).astype(float) / 255.0
            else:
                mask = mask.reshape((1, 1, height, width)).astype(float) / 255.0
            mask = torch.from_numpy(mask)
            mask = F.interpolate(mask, size=(self.opt.crop_size[0], self.opt.crop_size[1]),
                                 mode='bicubic', align_corners=False)[0]
            mask = mask.reshape(1, self.opt.crop_size[0], self.opt.crop_size[1]).type(torch.float)
            last_fake = 0
        return {'frame': frame, 'flow': flow, 'mask': mask, 'style': style, 'path': self.frames[video_id][frame_id],
                'last_fake': last_fake, 'last_fake_path': self.lasts_fake[video_id] if frame_id == 0 else '',
                'is_last': is_last}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return sum(self.lengths)
