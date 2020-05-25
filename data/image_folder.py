"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', 'pgm',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def make_test_video(dir_frame, dir_last_fake):
    frames = []
    lasts_fake = []
    assert os.path.isdir(dir_frame), '%s is not a valid directory' % dir_frame
    assert os.path.isdir(dir_last_fake), '%s is not a valid directory' % dir_last_fake
    for folder in sorted(os.listdir(dir_frame)):
        folder_path = os.path.join(dir_frame, folder)
        if os.path.isdir(folder_path):
            video = []
            for frame in os.listdir(folder_path):
                if is_image_file(frame):
                    path = os.path.join(folder_path, frame)
                    video.append(path)
            frames.append(sorted(video))
    for image in sorted(os.listdir(dir_last_fake)):
        if is_image_file(image):
            path = os.path.join(dir_last_fake, image)
            lasts_fake.append(path)
    return frames, lasts_fake


def make_video_dataset(dir_frame, dir_flow, dir_mask, dir_style, dir_last_fake):
    frames = []
    flows = []
    masks = []
    styles = []
    lasts_fake = []
    assert os.path.isdir(dir_frame), '%s is not a valid directory' % dir_frame
    assert os.path.isdir(dir_flow), '%s is not a valid directory' % dir_flow
    assert os.path.isdir(dir_mask), '%s is not a valid directory' % dir_mask
    assert os.path.isdir(dir_style), '%s is not a valid directory' % dir_style
    assert os.path.isdir(dir_last_fake), '%s is not a valid directory' % dir_last_fake
    for folder in sorted(os.listdir(dir_frame)):
        folder_path = os.path.join(dir_frame, folder)
        if os.path.isdir(folder_path):
            video = []
            for frame in os.listdir(folder_path):
                if is_image_file(frame):
                    path = os.path.join(folder_path, frame)
                    video.append(path)
            video.sort()
            video.reverse()
            frames.append(video)
    for folder in sorted(os.listdir(dir_flow)):
        folder_path = os.path.join(dir_flow, folder)
        if os.path.isdir(folder_path):
            video = []
            for flow in os.listdir(folder_path):
                if flow.endswith('.flo') or flow.endswith('.mat'):
                    path = os.path.join(folder_path, flow)
                    video.append(path)
            video.sort()
            video.reverse()
            flows.append(video)
    for folder in sorted(os.listdir(dir_mask)):
        folder_path = os.path.join(dir_mask, folder)
        if os.path.isdir(folder_path):
            video = []
            for mask in os.listdir(folder_path):
                if is_image_file(mask):
                    path = os.path.join(folder_path, mask)
                    video.append(path)
            video.sort()
            video.reverse()
            masks.append(video)
    for image in sorted(os.listdir(dir_style)):
        if is_image_file(image):
            path = os.path.join(dir_style, image)
            styles.append(path)
    for image in sorted(os.listdir(dir_last_fake)):
        if is_image_file(image):
            path = os.path.join(dir_last_fake, image)
            lasts_fake.append(path)
    return frames, flows, masks, styles, lasts_fake


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
