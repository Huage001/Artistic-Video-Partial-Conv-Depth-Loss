import os


dataroot = input('Please input your dataset root\n')
name = input('Please input your experiment name\n')
number = int(input('Please input the number of test image\n'))
load_size_w = int(input('Please input width of image load size\n'))
load_size_h = int(input('Please input height of image load size\n'))

os.system('python test.py --dataroot %s --name %s --model cycle_gan --no_flip --num_test %d --netG resnet_6blocks --load_size_w %d --load_size_h %d --crop_size_w %d --crop_size_h %d' % (dataroot, name, number, load_size_w, load_size_h, load_size_w, load_size_h))
