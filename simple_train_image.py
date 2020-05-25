import os


dataroot = input('Please input your dataset root\n')
name = input('Please input your experiment name\n')
epoch = int(input('Please input total number of training epoch\n'))
load_size_w = int(input('Please input width of image load size\n'))
load_size_h = int(input('Please input height of image load size\n'))
crop_size_w = int(input('Please input width of image crop size\n'))
crop_size_h = int(input('Please input height of image crop size\n'))
batch_size = int(input('Please input batch size\n'))

os.system('python train.py --dataroot %s --name %s --model cycle_gan --niter %d --niter_decay %d --load_size_w %d --load_size_h %d --crop_size_w %d --crop_size_h %d --netG resnet_6blocks --batch_size %d' % (dataroot, name, epoch // 2, epoch // 2, load_size_w, load_size_h, crop_size_w, crop_size_h, batch_size))

