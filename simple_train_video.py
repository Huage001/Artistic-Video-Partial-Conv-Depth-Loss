import os


dataroot = input('Please input your dataset root\n')
name = input('Please input your experiment name\n')
epoch = int(input('Please input total number of training epoch\n'))
load_size_w = int(input('Please input width of image load size\n'))
load_size_h = int(input('Please input height of image load size\n'))
prev_name = input('Please input experiment name of stage 1\n')

os.system('python train.py --dataroot %s --name %s --model video --no_flip --niter 0 --niter_decay 0 --load_size_w %d --load_size_h %d --crop_size_w %d --crop_size_h %d --netG resnet_6blocks'% (dataroot, name, load_size_w, load_size_h, load_size_w, load_size_h))
os.system('cp %s %s' % ('checkpoints/' + prev_name + '/latest_net_G_B_encoder.pth', 'checkpoints/' + name))
os.system('cp %s %s' % ('checkpoints/' + prev_name + '/latest_net_G_B_encoder.pth', 'checkpoints/' + name))
os.system('cp %s %s' % ('checkpoints/' + prev_name + '/latest_net_G_A.pth', 'checkpoints/' + name))
os.system('cp %s %s' % ('checkpoints/' + prev_name + '/latest_net_D_B.pth', 'checkpoints/' + name))
os.system('python train.py --dataroot %s --name %s --model video --no_flip --niter %d --niter_decay %d --load_size_w %d --load_size_h %d --crop_size_w %d --crop_size_h %d --netG resnet_6blocks --continue_train' % (dataroot, name, epoch // 2, epoch // 2, load_size_w, load_size_h, load_size_w, load_size_h))

