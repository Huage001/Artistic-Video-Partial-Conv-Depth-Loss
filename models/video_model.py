import torch
import itertools
import numpy as np
import torch.nn.functional as F
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import util


def recursion_fix_version(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
        module.num_batches_tracked = 1
    elif isinstance(module, torch.nn.AvgPool2d):
        module.divisor_override = None
    elif isinstance(module, torch.nn.UpsamplingNearest2d):
        module.mode = 'nearest'
        module.align_corners = None
        module.size = None
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_fix_version(module1)
    return module


class VideoModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_depth', type=float, default=16.0,
                                help='use depth loss to hold semantic consistency')
            parser.add_argument('--depth_path', type=str, default='./models/depth_model.pt')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_video', type=float, default=16.0, help='weight for optical flow loss')
            parser.add_argument('--lambda_var', type=float, default=8.0, help='weight for variance loss')
            parser.add_argument('--bptt', type=int, default=2, help='steps of backpropagation')

        parser.add_argument('--adv_mask', action='store_true',
                            help='if 0 in the video mask denotes need optical flow, please set this true')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_B', 'G_B', 'cycle_B', 'video', 'depth_B', 'var']
        if self.isTrain:
            self.visual_names = ['prev_fake_A', 'warp_fake_A', 'real_B', 'fake_A', 'rec_B']
        else:
            self.visual_names = ['fake_A']

        if self.isTrain:
            self.model_names = ['G_A', 'G_B_encoder', 'G_B_decoder', 'G_Warp_encoder']
            self.model_names += ['D_B']
        else:
            self.model_names = ['G_B_encoder', 'G_B_decoder', 'G_Warp_encoder']

        # define networks (both Generators and discriminators)
        self.netG_B_encoder, self.netG_B_decoder = networks.define_Coder(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, partial_conv=False)
        self.netG_Warp_encoder = networks.define_Coder(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, only_encoder=True, partial_conv=True)

        if self.isTrain:
            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain and self.opt.lambda_depth > 0.0:
            self.net_depth = torch.load(self.opt.depth_path)
            for i, (name, module) in enumerate(self.net_depth._modules.items()):
                module = recursion_fix_version(self.net_depth)
            self.net_depth = self.net_depth.to(self.device)
            self.set_requires_grad([self.net_depth], False)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            self.criterionVideo = torch.nn.MSELoss().to(self.device)
            self.criterionDepth = torch.nn.MSELoss().to(self.device)
            self.criterionVar = networks.VarLoss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B_encoder.parameters(),
                                                                self.netG_B_decoder.parameters(),
                                                                self.netG_Warp_encoder.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.isFirst = False
        self.flow = input['flow'].to(self.device) if input['flow'].dim() == 4 else None
        self.mask = input['mask'].to(self.device) if input['mask'].dim() == 4 else None
        self.image_paths = input['path']
        self.prev_fake_path = input['last_fake_path']
        self.real_B = input['frame'].to(self.device)
        self.real_A = input['style'].to(self.device)
        if self.flow is None:
            self.isFirst = True
            self.prev_fake_A = input['last_fake'].to(self.device)
        else:
            self.prev_fake_A = self.fake_A.detach()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isFirst:
            self.warp_fake_A = self.prev_fake_A
            self.mask = torch.ones(1, 1, self.opt.crop_size[0], self.opt.crop_size[1]).to(self.device)
        else:
            self.warp_fake_A = F.grid_sample(self.prev_fake_A, self.flow, align_corners=True) * self.mask
        fake_A_code = self.netG_B_encoder(self.real_B)
        warp_A_code, mask = self.netG_Warp_encoder(self.warp_fake_A, self.mask)
        mask = mask.repeat(1, warp_A_code.shape[1], 1, 1)
        self.fake_A = self.netG_B_decoder(torch.cat((fake_A_code, warp_A_code), dim=1),
                                          torch.cat((torch.ones_like(mask), mask), dim=1))
        if self.isTrain:
            self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_B = self.opt.lambda_B
        lambda_video = self.opt.lambda_video
        lambda_depth = self.opt.lambda_depth
        lambda_var = self.opt.lambda_var

        if lambda_depth > 0:
            self.depth_real_B = self.net_depth(self.real_B)
            self.depth_fake_A = self.net_depth(self.fake_A)
            instance_norm = torch.nn.InstanceNorm2d(3)
            self.loss_depth_B = self.criterionDepth(instance_norm(self.depth_real_B),
                                                    instance_norm(self.depth_fake_A)) * lambda_depth
        else:
            self.loss_depth_B = 0.0

        if lambda_var > 0:
            self.loss_var = self.criterionVar(self.fake_A) * lambda_var
        else:
            self.loss_var = 0.0

        if self.isFirst:
            self.loss_video = 0
        else:
            self.loss_video = self.criterionVideo(self.warp_fake_A * self.mask,
                                                  self.fake_A * self.mask) * lambda_video

        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        if self.opt.bptt > 1:
            if self.isFirst:
                self.loss_G_B_cache = []
            self.loss_G_B_cache.append(self.loss_G_B + self.loss_cycle_B + self.loss_video + self.loss_depth_B)
            if len(self.loss_G_B_cache) > self.opt.bptt:
                del self.loss_G_B_cache[0]
            self.loss_G = sum(self.loss_G_B_cache) / len(self.loss_G_B_cache)
            self.loss_G.backward(retain_graph=True)
        else:
            self.loss_G = self.loss_G_B + self.loss_cycle_B + self.loss_video + self.loss_depth_B
            self.loss_G.backward()


    def optimize_parameters(self):
        self.set_requires_grad([self.netG_A], False)
        self.forward()
        self.set_requires_grad([self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_B()
        self.optimizer_D.step()
        if self.isFirst:
            im_fake = util.tensor2im(self.fake_A.detach())
            util.save_image(im_fake, self.prev_fake_path[0])
