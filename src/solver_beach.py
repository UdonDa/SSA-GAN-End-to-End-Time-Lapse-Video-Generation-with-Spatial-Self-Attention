from sys import exit
import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import time
import itertools
from torchvision.utils import save_image
import eval_metrics as eval_metrics
from tqdm import tqdm
import pytorch_ssim

from utils import denorm, weights_init, generate_video


from models.model_RefineNet import RefineNetGenerator, RefineNetDiscriminator
from PIL import Image

class Solver(object):
    def __init__(self, train_loader, eval_loader, config):
        """Initialize configurations."""
        self.config = config
        # Data Loader.
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.eval_iter = iter(eval_loader)

        # Model hyper-parameters.
        self.image_size = config.image_size
        self.lambda_cycle = config.lambda_cycle
        self.lambda_gp = config.lambda_gp
        self.lambda_triplet = config.lambda_triplet

        # Training setting.
        self.batch_size = config.batch_size
        self.base_g_lr = config.g_lr
        self.base_d_lr = config.d_lr
        self.refine_g_lr = config.g_lr
        self.refine_d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.n_critic = config.n_critic
        self.stage1 = config.stage1
        self.stage2 = config.stage2
        self.stage1_resume_iter = config.stage1_resume_iter
        self.stage2_resume_iter = config.stage2_resume_iter

        # Pretrained model.
        self.pretrained_model = config.pretrained_model

        # Misc.
        self.train_mode = config.train
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resume_iters = config.resume_iters
        self.nframes = config.nframes
        self.metrics = ['ssim', 'psnr', 'mse']

        # Path.
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.image_path = config.image_path
        self.gif_path = config.gif_path
        self.val_path = config.val_path
        self.metric_path = config.metric_path

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.eval_step = config.eval_step

        # if config.mode == "train":
        self.build_model()
        # else:
        #     self.build_eval_model()
        if config.mode == "train":
            self.build_tensorboard()

        self.ssim_loss = pytorch_ssim.SSIM()


    def build_model(self):
        """Create a generator and a discriminators."""
        if self.stage1:
            if self.config.my_model:
                if self.config.channel_cat:
                    if self.config.image_size == 128:
                        print('channel cat model 128')
                        # noGammaHW does not work...
                        # Please read our paper of Sec 5.4(Ablation study).
                        if self.config.attention_type == "HW" or self.config.attention_type == "noGammaHW":
                            from models.Dynamic_Model_attn_channel_cat import MDGAN_S1_G, MDGAN_S2_G, MDGAN_S2_D
                        if self.config.attention_type == "THW" or self.config.attention_type == "noGammaTHW":
                            from models.Dynamic_Model_attn_channel_cat_thw import MDGAN_S1_G, MDGAN_S2_G, MDGAN_S2_D
                    elif self.config.image_size == 64:
                        print('channel cat model 64')
                        from models.Dynamic_Model_attn_channel_cat_64 import MDGAN_S1_G, MDGAN_S2_G, MDGAN_S2_D

                else:
                    if self.config.image_size == 128:
                        from models.Dynamic_Model_attn import MDGAN_S1_G, MDGAN_S2_G, MDGAN_S2_D
                    elif self.config.image_size == 64:
                        from models.Dynamic_Model_attn_64 import MDGAN_S1_G, MDGAN_S2_G, MDGAN_S2_D
                    else:
                        print('wrong image_size: {}'.format(self.config.image_size))
            else:
                if self.config.image_size == 128:
                    print('Use MD-GAN Model 128')
                    from models.Dynamic_Model import MDGAN_S1_G, MDGAN_S2_G, MDGAN_S2_D
                elif self.config.image_size == 64:
                    print('Use MD-GAN Model 64')
                    from models.Dynamic_Model_64 import MDGAN_S1_G, MDGAN_S2_G, MDGAN_S2_D


            if self.config.my_model:
                self.BaseG = nn.DataParallel(MDGAN_S1_G(sn=self.config.use_spectral_norm_G, ngf=32, args=self.config).to(self.device))
            else:
                self.BaseG = nn.DataParallel(MDGAN_S1_G(sn=self.config.use_spectral_norm_G, ngf=32).to(self.device))
            self.BaseD = nn.DataParallel(MDGAN_S2_D(sn=self.config.use_spectral_norm_D, ndf=32).to(self.device))

            self.BaseG.apply(weights_init)
            self.BaseD.apply(weights_init)

            self.base_g_optimizer = torch.optim.Adam(self.BaseG.parameters(), self.base_g_lr, [self.beta1, self.beta2])
            self.base_d_optimizer = torch.optim.Adam(self.BaseD.parameters(), self.base_d_lr, [self.beta1, self.beta2])

            self.base_g_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.base_g_optimizer, milestones=[45, 55], gamma=0.1)
            self.base_d_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.base_d_optimizer, milestones=[45, 55], gamma=0.1)


        
        if self.stage2:
            if self.config.my_model:
                if self.config.image_size == 128:
                    print('Use OWN-Model 128 in stage 2.')
                    from models.Dynamic_Model_attn import MDGAN_S2_G
                    from models.Dynamic_Model_attn_channel_cat import MDGAN_S1_G, MDGAN_S2_D
                elif self.config.image_size == 64:
                    print('Use OWN-Model 64 in stage 2.')
                    from models.Dynamic_Model_attn_64 import MDGAN_S2_G
                    from models.Dynamic_Model_attn_channel_cat_64 import MDGAN_S1_G, MDGAN_S2_D
                else:
                    print('wrong image_size: {}'.format(self.config.image_size))
                
            else:
                if self.config.image_size == 128:
                    print('Use MD-GAN Model 128 in stage 2')
                    from models.Dynamic_Model import MDGAN_S1_G, MDGAN_S2_G, MDGAN_S2_D
                elif self.config.image_size == 64:
                    print('Use MD-GAN Model 64 in stage 2')
                    from models.Dynamic_Model_64 import MDGAN_S1_G, MDGAN_S2_G, MDGAN_S2_D
                

            if self.config.my_model:
                self.BaseG = MDGAN_S1_G(sn=self.config.use_spectral_norm_G, ngf=32, args=self.config).to(self.device)
            else:
                self.BaseG = MDGAN_S1_G(sn=self.config.use_spectral_norm_G, ngf=32).to(self.device)
            
            # for param in self.BaseG.parameters():
            #     param.requires_grad = False
            self.BaseG.eval()
            self.BaseG = nn.DataParallel(self.BaseG)
            self.RefineG = nn.DataParallel(MDGAN_S2_G(sn=self.config.use_spectral_norm_G).to(self.device))
            self.RefineD = nn.DataParallel(MDGAN_S2_D(sn=self.config.use_spectral_norm_D).to(self.device))
            self.base_g_optimizer = torch.optim.Adam(self.BaseG.parameters(), self.base_g_lr, [self.beta1, self.beta2])
            self.refine_g_optimizer = torch.optim.Adam(self.RefineG.parameters(), self.base_g_lr, [self.beta1, self.beta2])
            self.refine_d_optimizer = torch.optim.Adam(self.RefineD.parameters(), self.base_d_lr, [self.beta1, self.beta2])

            self.refine_g_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.refine_g_optimizer, milestones=[30, 40, 50], gamma=0.1)
            self.refine_d_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.refine_d_optimizer, milestones=[30, 40, 50], gamma=0.1)

    def print_network(self, model):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, model_path):
        print('Loading the trained base models from \n{}'.format(model_path))
        G_path = os.path.join(model_path)

        state_dict = torch.load(G_path, map_location=lambda storage, loc: storage)

        # # create new OrderedDict that does not contain `module.`
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        self.BaseG.load_state_dict(state_dict)


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from tensorboardX import SummaryWriter
        self.logger = SummaryWriter(self.log_path)


    def reset_grad(self):
        if not self.stage2:
            self.base_g_optimizer.zero_grad()
            self.base_d_optimizer.zero_grad()
        elif self.stage2:
            self.refine_g_optimizer.zero_grad()
            self.refine_g_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty_v2(self, x, y, f):
        # interpolation
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        alpha = torch.rand(shape).to(self.device)
        z = x + alpha * (y - x)

        # gradient penalty
        z = torch.autograd.Variable(z, requires_grad=True).to(self.device)
        o = f(z)[0]
        g = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size()).to(self.device), create_graph=True)[0].view(z.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

        return gp

    def compute_gram_matrix(self, x):
        """
        Input: x -> [N, C, T, H, W]
        論文より，x_hat -> [N, M, S]に変形してGramMatrixを計算.
        (M = C * T, S = H * W)

        The Gram matrices are calculated using the features
        of the first and third conv layers(after ReLU layer) od D_2.(p6)
                -> 1, 3番目の特徴量で計算する.

        We employ the Gram matrix [4] as the motion feature
        representation to assist G2 to learn dynamics across video
        frames.
        """
        n, c, t, h, w = x.size()
        M = c * t
        S = h * w

        x = x.view(n, c, t, -1)
        x = x.view(n, -1, x.size()[-1]) # -> [N, 96, 16384]

        x_t = x.permute(0,2,1)

        loss = torch.bmm(x, x_t) / (S * M * n)


        # for batch in range(n):
        #     f = x[batch] # -> [96, 16384]
        #     f_t = torch.t(f) # -> [16384, 96]
        #     gram_matrix = torch.mm(f, f_t) # -> [n*c, n*c]
        #     loss += gram_matrix
        # loss /= (M * S)
        return loss.mean()


    def compute_ranking_loss(self, anchor=None, negative=None, positive=None):
        loss = 0
        for i in range(2):
            y_1 = self.compute_gram_matrix(anchor[i])
            y_2 = self.compute_gram_matrix(negative[i])
            y = self.compute_gram_matrix(positive[i])
            A = torch.abs(y_1 - y)
            denominator = A.exponential_(1)
            B = torch.abs(y_2 - y)
            numerator = B.exponential_(1) + A.exponential_(1)
            fraction = torch.mean((-1) * torch.log(denominator / numerator))
            loss = loss + fraction
        return loss


    # def compute_triplet_loss(self, anchor=None, negative=None, positive=None):
    #     loss = 0
    #     for i in range(2):
    #         A = - torch.abs(anchor[i] - positive[i])
    #         denominator = A.exponential_(1).mean()
    #         # print('denominator: ', denominator)
    #         B = - torch.abs(anchor[i] - negative[i])
    #         numerator = (B.exponential_(1) + A.exponential_(1)).mean()
    #         # print('numerator: ', numerator)

    #         fraction = (- torch.log(denominator / numerator)).mean()

    #         # print('torch.log(denominator / numerator): ', torch.log(denominator / numerator).mean())
    #         # print('fraction: ', fraction.size())
    #         loss = loss + fraction
    #     return loss


    def define_metrics(self):
        # Define metrics.
        if self.metrics is not None:
            self.metrics_values = {}
            for metric_name in self.metrics:
                self.metrics_values['{}_frames'.format(metric_name)] = torch.zeros_like(torch.FloatTensor(len(self.eval_loader) * self.batch_size, self.nframes))
                self.metrics_values['{}_avg'.format(metric_name)] = torch.zeros_like(torch.FloatTensor(len(self.eval_loader) * self.batch_size,1))

        if self.metrics is not None:
            self.metrics_i_video = {}
            for metric_name in self.metrics:
                self.metrics_i_video['{}_i_video'.format(metric_name)] = 0

    
    def eval(self, epoch):
        self.define_metrics()

        with torch.no_grad():
            for i, (frames, _) in enumerate(self.eval_loader):
                real_frames = frames.to(self.device)
                input_frames = real_frames[:,:,0:1,:,:]
                input_frames = input_frames.repeat(1, 1, self.nframes, 1, 1)
                
                fake_frames = self.BaseG(input_frames.detach())

                if self.stage2:
                    self.RefineG.eval()
                    first_stage_frames = fake_frames.detach()
                    fake_frames = self.RefineG(fake_frames.detach())

                if self.metrics is not None:
                    for metric_name in self.metrics:
                        calculate_metric = getattr(eval_metrics, 'calculate_{}'.format(metric_name))

                        for i_batch in range(fake_frames.size(0)):
                            for i_frame in range(self.nframes):
                                self.metrics_values['{}_frames'.format(metric_name)][self.metrics_i_video['{}_i_video'.format(metric_name)], i_frame] = calculate_metric(fake_frames[i_batch,:,i_frame,:,:], real_frames[i_batch,:,i_frame,:,:])
                                self.metrics_values['{}_avg'.format(metric_name)][self.metrics_i_video['{}_i_video'.format(metric_name)]] = torch.mean(self.metrics_values['{}_frames'.format(metric_name)][self.metrics_i_video['{}_i_video'.format(metric_name)]])
                            self.metrics_i_video['{}_i_video'.format(metric_name)] = self.metrics_i_video['{}_i_video'.format(metric_name)] + 1
                #             break # TODO: debug
                #         break # TODO: debug
                # break # TODO: debug

                if i % 5 == 0:
                    image_path = os.path.join(self.image_path, str(epoch), str(i))
                    os.makedirs(image_path, exist_ok=True)
                    if not self.stage2:
                        frames = torch.cat([real_frames, fake_frames], dim=4)
                    else:
                        frames = torch.cat([real_frames, first_stage_frames, fake_frames], dim=4)
                    for j in range(self.nframes):
                        a = frames[0][0][j].unsqueeze(0)
                        b = frames[0][1][j].unsqueeze(0)
                        c = frames[0][2][j].unsqueeze(0)
                        A = torch.cat([a, b, c], dim=0)
                        save_image(denorm(A.data.cpu()), f'{image_path}/{str(j).zfill(5)}.png')
                    
                    generate_video(
                        inputs=image_path,
                        output_dir=self.gif_path,
                        filename=f"{epoch}_{i}"
                    )

        # return    
        # calculate and save mean eval statistics
        if self.metrics is not None:
            metrics_mean_values = {}
            for metric_name in self.metrics:
                test_dir = self.metric_path

                metrics_mean_values['{}_frames'.format(metric_name)] = torch.mean(self.metrics_values['{}_frames'.format(metric_name)],0)
                metrics_mean_values['{}_avg'.format(metric_name)] = torch.mean(self.metrics_values['{}_avg'.format(metric_name)],0)
                # torch.save(metrics_mean_values['{}_frames'.format(metric_name)], os.path.join(test_dir, '{}_frames.pt'.format(metric_name)))
                # torch.save(metrics_mean_values['{}_avg'.format(metric_name)], os.path.join(test_dir, '{}_avg.pt'.format(metric_name)))

            # print(metrics_mean_values)
            with open('{}/eval_result_{}.txt'.format(self.metric_path, epoch), 'w') as f:
                print(metrics_mean_values, file=f)
                self.logger.add_scalar(f'ssim_avg', metrics_mean_values['ssim_avg'], epoch)
                self.logger.add_scalar(f'psnr_avg', metrics_mean_values['psnr_avg'], epoch)
                self.logger.add_scalar(f'mse_avg', metrics_mean_values['mse_avg'], epoch)


    def train_stage1(self):
        train_loader = self.train_loader

        print('Start training...')
        loss = {}
        count = 0
        for epoch in range(1, self.config.total_epochs+1):
            self.base_g_scheduler.step()
            self.base_d_scheduler.step()

            p_bar = tqdm(train_loader)
            for i, (real_frames, _) in enumerate(p_bar):
                real_frames = real_frames.to(self.device)
                first_frames = real_frames[:,:,0:1,:,:]
                input_frames = first_frames.repeat(1,1,self.nframes,1,1)

                ### ================================== Train Discriminator. ================================
                out_real, attn_real = self.BaseD(real_frames)
                d_loss_real = - out_real.mean()
                fake_frames = self.BaseG(input_frames)
                out_fake, attn_fake = self.BaseD(fake_frames)
                d_loss_fake = out_fake.mean()
                d_loss_gp = self.gradient_penalty_v2(real_frames.data, fake_frames.data, self.BaseD)

                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
                
                self.reset_grad()
                d_loss.backward()
                self.base_d_optimizer.step()

                loss['D/real'] = d_loss_real.item()
                loss['D/fake'] = d_loss_fake.item()
                loss['D/gp'] = d_loss_gp.item()
                loss['D/loss'] = d_loss.item()

                if (i+1) % self.n_critic == 0:
                    ### ===================================== Train generator. ==============================
                    fake_frames = self.BaseG(input_frames)
                    out_fake = self.BaseD(fake_frames)[0]
                    g_loss_fake = - out_fake.mean()
                    g_loss_L1 = torch.abs(fake_frames - real_frames).mean()

                    # SSIM Loss
                    if self.config.my_model:
                        g_loss_ssim = 1 - self.ssim_loss(real_frames, fake_frames)
                        g_loss = g_loss_fake + g_loss_L1 + g_loss_ssim
                        loss['G/ssim'] = g_loss_ssim.item()
                    else:
                        g_loss = g_loss_fake + g_loss_L1
                    
                    self.reset_grad()
                    g_loss.backward()
                    self.base_g_optimizer.step()
                    loss['G/fake'] = g_loss_fake.item()
                    # if self.config.my_model:
                    loss['G/L1'] = g_loss_L1.item()
                    loss['G/loss'] = g_loss.item()
                

                log = "Epoch[{}/{}]".format(epoch, self.config.total_epochs+1)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)

            
            if epoch % 2 == 0:
                print('=========== Eval step. ===========')
                self.eval(epoch)

            if epoch % self.model_save_step == 0:
                print('=========== Save models step. =============')
                torch.save(self.BaseG.state_dict(), os.path.join(self.model_save_path, '{}-BaseG.ckpt'.format(epoch)))
                torch.save(self.BaseD.state_dict(), os.path.join(self.model_save_path, '{}-BaseD.ckpt'.format(epoch)))

    
    def train_stage2(self):
        train_loader = self.train_loader
        print('Start training stage2...')
        self.restore_model(self.config.pretrained_model_path)

        loss = {}
        start_time = time.time()
        count = 0
        
        for epoch in range(self.config.total_epochs):
            self.BaseG.eval()
            self.RefineG.train()
            self.RefineD.train()
            self.refine_g_scheduler.step()
            self.refine_d_scheduler.step()

            p_bar = tqdm(train_loader)
            for i, (real_frames, _) in enumerate(p_bar):
                real_frames = real_frames.to(self.device) # -> [B, 3, 32, 128, 128]
                first_frames = real_frames[:,:,0:1,:,:] # -> [2, 3, 128, 128]
                input_frames = first_frames.repeat(1,1,self.nframes,1,1)
        
                ### ================================== Train Discriminator. ================================
                
                out_real, out_f_real = self.RefineD(real_frames)
                d_loss_real = - out_real.mean()
                
                with torch.no_grad():
                    fake_base_frames = self.BaseG(input_frames.detach())
                fake_frames = self.RefineG(fake_base_frames)
                out_fake, out_f_fake_refine = self.RefineD(fake_frames)
                d_loss_fake = out_fake.mean()
                d_loss_gp = self.gradient_penalty_v2(real_frames.data, fake_frames.data, self.RefineD)

                with torch.no_grad():
                    _, out_f_fake_base = self.RefineD(fake_base_frames)

                # d_loss_triplet = self.compute_triplet_loss(anchor=out_f_fake_refine, negative=out_f_fake_base, positive=out_f_real)
                d_loss_triplet = self.compute_ranking_loss(anchor=out_f_fake_refine, negative=out_f_fake_base, positive=out_f_real)
                

                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp + self.lambda_triplet * d_loss_triplet

                self.reset_grad()
                d_loss.backward()
                self.refine_d_optimizer.step()

                loss['D/triplet'] = d_loss_triplet.item() * self.lambda_triplet
                loss['D/real'] = d_loss_real.item()
                loss['D/fake'] = d_loss_fake.item()
                loss['D/gp'] = d_loss_gp.item() * self.lambda_gp
                loss['D/loss'] = d_loss.item()

                if (i+1) % self.n_critic == 0:
                    ### ===================================== Train generator. ==============================
                    fake_frames = self.RefineG(fake_base_frames)
                    out_fake = self.RefineD(fake_frames)[0]
                    g_loss_fake = - out_fake.mean()
                    g_loss_L1 = torch.abs(fake_frames - real_frames).mean()

                    g_loss = g_loss_fake + g_loss_L1
                    self.reset_grad()
                    g_loss.backward()
                    self.refine_g_optimizer.step()

                    loss['G/fake'] = g_loss_fake.item()
                    loss['G/L1'] = g_loss_L1.item()
                    loss['G/loss'] = g_loss.item()

                log = "Epoch[{}/{}]".format(epoch, self.config.total_epochs+1)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)

                p_bar.set_description(log)
                for tag, value in loss.items():
                    self.logger.add_scalar(tag, value, count)
                count += 1


            if epoch % 2 == 0:
                print('=========== Eval step. ===========')
                self.eval(epoch)

            if epoch % self.model_save_step == 0:
                print("=========== Save models step. =============")
                torch.save(self.RefineG.state_dict(), os.path.join(self.model_save_path, '{}-RefineG.ckpt'.format(epoch)))
                torch.save(self.RefineD.state_dict(), os.path.join(self.model_save_path, '{}-RefineD.ckpt'.format(epoch)))

    def build_eval_model(self):
        """Create a generator and a discriminators."""
    
        ### own
        if self.config.image_size == 128:
            print('Use OWN-Model 128 in stage 2.')
            from models.Dynamic_Model_attn import MDGAN_S2_G as OWN_S2_G
            from models.Dynamic_Model_attn_channel_cat import MDGAN_S1_G as OWN_S1_G
        elif self.config.image_size == 64:
            print('Use OWN-Model 64 in stage 2.')
            from models.Dynamic_Model_attn_64 import MDGAN_S2_G as OWN_S2_G
            from models.Dynamic_Model_attn_channel_cat_64 import MDGAN_S1_G  as OWN_S1_G

        self.OwnBaseG = nn.DataParallel(OWN_S1_G(sn=True, ngf=32).to(self.device))
        self.OwnRefineG = nn.DataParallel(OWN_S2_G(sn=True, ngf=32).to(self.device))

        ### md
        if self.config.image_size == 128:
            print('Use MD-GAN Model 128 in stage 2')
            from models.Dynamic_Model import MDGAN_S1_G, MDGAN_S2_G, MDGAN_S2_D
        elif self.config.image_size == 64:
            print('Use MD-GAN Model 64 in stage 2')
            from models.Dynamic_Model_64 import MDGAN_S1_G, MDGAN_S2_G, MDGAN_S2_D

        self.MdBaseG = nn.DataParallel(MDGAN_S1_G(sn=False, ngf=32).to(self.device))
        self.MdRefineG = nn.DataParallel(MDGAN_S2_G(sn=False, ngf=32).to(self.device))

    def save_beach_imgs(self):
        
        img_path = '{}/img'.format(self.config.exp_name)
        mp4_path = '{}/mp4'.format(self.config.exp_name)

        if not os.path.exists(img_path):
            os.makedirs(img_path)
        if not os.path.exists(mp4_path):
            os.makedirs(mp4_path)

        def saveer(frames, base_path):
            # print('Save !!!!')
            for j in range(32):
                A = frames[:,:,j,:,:]
                save_image(denorm(A.data.cpu()), '{}/sample_{}.png'.format(base_path, str(j+1).zfill(3)))
            images = []

        def restore_pretrained_model(path, model, flag=False):
            if flag:
                state_dict = torch.load(path, map_location=lambda storage, loc: storage)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                new_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
                model.load_state_dict(new_state_dict)
                return model
            else:
                new_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
                model.load_state_dict(new_state_dict)
                return model

        eval_loader = self.eval_loader
        print("start model loading...")
        self.OwnBaseG = restore_pretrained_model(self.config.own_stage_1, self.OwnBaseG)
        self.OwnBaseG.eval()
        print("own base g")
        self.OwnRefineG = restore_pretrained_model(self.config.own_stage_2, self.OwnRefineG)
        self.OwnRefineG.eval()
        print("own refine g")
        self.MdBaseG = restore_pretrained_model(self.config.md_stage_1, self.MdBaseG, flag=True)
        self.MdBaseG.eval()
        print("md base g")
        self.MdBaseG = restore_pretrained_model(self.config.md_stage_2, self.MdRefineG, flag=True)
        self.MdBaseG.eval()
        print("md refine g")


        def generate_video(img_path, out_path):
            img_path = img_path + '/sample_%03d.png'
            mp4_path = out_path
            cmd = ('ffmpeg -loglevel warning -framerate 25 -i ' + img_path + 
                ' -vcodec libx264 -pix_fmt yuv420p -qscale:v 2 -y ' + mp4_path )
            print(cmd)
            os.system(cmd)

        counter = 0
        print('Start save beach images...')
        for i, (real_frames, _) in enumerate(eval_loader):
            out_path = "{}/{}".format(self.config.exp_name, i)
            # if not os.path.exists(out_path):
            #     os.makedirs(out_path)

            real_frames = real_frames.to(self.device) # -> [B, 3, 32, 128, 128]
            first_frames = real_frames[:,:,0:1,:,:] # -> [2, 3, 128, 128]
            input_frames = first_frames.repeat(1,1,self.nframes,1,1)


            val_fake_own_s1 = self.OwnBaseG(input_frames)
            val_fake_own_s2 = self.OwnRefineG(input_frames)
            val_fake_md_s1 = self.MdBaseG(input_frames)
            val_fake_md_s2 = self.MdRefineG(input_frames)

            concated_frames = torch.cat([real_frames.to(self.device), 
                val_fake_own_s1, val_fake_own_s2, val_fake_md_s1, val_fake_md_s2], dim=4)

            for i_batch in range(concated_frames.size(0)):
                frames = concated_frames[i_batch:i_batch+1,:,:,:,:]
                save_img_path = "{}/{}".format(img_path, str(counter).zfill(3))
                if not os.path.exists(save_img_path):
                    os.makedirs(save_img_path)
                saveer(frames, save_img_path)
                
                save_mp4_path = "{}/{}.mp4".format(mp4_path, counter)
                # if not os.path.exists(save_mp4_path):
                #     os.makedirs(save_mp4_path)
                generate_video(save_img_path, save_mp4_path)



                counter += 1

            if counter % 20 == 0:
                print('saved {} videos.'.format(counter * self.batch_size))

    def test(self, loader):
        print("Validation start.")
        self.restore_model(self.config.pretrained_model_path)
        SAVE_DIR = self.config.test_generated_path
        os.makedirs(f"{SAVE_DIR}/gifs", exist_ok=True)
        print(f"Save dir... {SAVE_DIR}")

        with torch.no_grad():
            for i, (real_frames, _) in enumerate(tqdm(loader)):
                image_path = f"{SAVE_DIR}/{i}"
                os.makedirs(image_path, exist_ok=True)
                real_frames = real_frames.cuda()
                real_frames = real_frames.unsqueeze(2)
                
                first_frames = real_frames[:,:,0:1,:,:]
                input_frames = first_frames.repeat(1,1,self.nframes,1,1)

                fake_frames = self.BaseG(input_frames)

                for j in range(self.nframes):
                    a = fake_frames[0][0][j].unsqueeze(0)
                    b = fake_frames[0][1][j].unsqueeze(0)
                    c = fake_frames[0][2][j].unsqueeze(0)
                    A = torch.cat([a, b, c], dim=0)
                
                    save_image(self.denorm(A.data.cpu()), f'{image_path}/{str(j).zfill(5)}.png')

                generate_video(
                    inputs=image_path,
                    output_dir=f"{SAVE_DIR}/gifs",
                    filename=f"{i}"
                )


    def generate_val_samples(self, loader):
        print("Start generate_val_samples mode.")
        self.restore_model(self.config.pretrained_model_path)
        from models.Dynamic_Model_attn_channel_cat import MDGAN_S1_G as Own_S1
        from models.Dynamic_Model_attn_channel_cat import MDGAN_S2_G as Own_S2
        # from models.Dynamic_Model import MDGAN_S1_G as MD_S1
        # from models.Dynamic_Model import MDGAN_S2_G as MD_S2
        from models.Dynamic_Model_original import MDGAN_S1_G as MD_S1
        from models.Dynamic_Model_original import MDGAN_S2_G as MD_S2

        BASE = "/home/yanai-lab/horita-d/beach_udon_net/results_cvpr/"

        own_s1 = nn.DataParallel(Own_S1(sn=self.config.use_spectral_norm_G, ngf=32, args=self.config).to(self.device))
        G_path = os.path.join(BASE, "7_1_WH_own_S1_L1-10/models/80-BaseG.ckpt")
        own_s1.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        own_s2 = nn.DataParallel(Own_S2(sn=self.config.use_spectral_norm_G, ngf=32).to(self.device))
        G_path = os.path.join(BASE, "7_22_own_S2/models/80-RefineG.ckpt")
        own_s2.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        self.config.attention_type = "noGammaHW"
        own_s1_noy = nn.DataParallel(Own_S1(sn=self.config.use_spectral_norm_G, ngf=32, args=self.config).to(self.device))
        G_path = os.path.join(BASE, "7_1_noGammaHW_own_S1_L1-10/models/80-BaseG.ckpt")
        own_s1_noy.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        # md_s1 = nn.DataParallel(MD_S1(sn=False, ngf=32).to(self.device))
        # G_path = os.path.join(BASE, "7_1_MD_S1/models/40-BaseG.ckpt")
        # md_s1.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        # md_s2 = nn.DataParallel(MD_S2(sn=True, ngf=32).to(self.device))
        # G_path = os.path.join(BASE, "7_22_MD_S2/models/70-RefineG.ckpt")
        # md_s2.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        # md_s1 = MD_S1(32).to(self.device)
        # G_path = os.path.join(BASE, "../wei_pretrained_model/netG_S1_030.pth")
        # md_s1.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        # md_s1 = nn.DataParallel(md_s1)

        # md_s2 = MD_S2(32).to(self.device)
        # G_path = os.path.join(BASE, "../wei_pretrained_model/netG_S2_067.pth")
        # md_s2.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        # md_s2 = nn.DataParallel(md_s2)



        # SAVE_DIR = "./samples_wei"
        # SAVE_DIR = self.config.test_generated_path
        SAVE_DIR = "./samples_ablation"
        os.makedirs(SAVE_DIR, exist_ok=True)

        os.makedirs(f"{SAVE_DIR}/gifs", exist_ok=True)
        print(f"Save dir... {SAVE_DIR}")
        used_batch = 0
        with torch.no_grad():
            p_bar = tqdm(loader)
            for i, (real_frames, _) in enumerate(p_bar):
                
                real_frames = real_frames.cuda()
                
                first_frames = real_frames[:,:,0:1,:,:]
                input_frames = first_frames.repeat(1,1,self.nframes,1,1)

                fake_owns1 = own_s1(input_frames)
                # fake_owns2 = own_s2(fake_owns1)
                fake_owns_noy = own_s1_noy(input_frames)

                # fake_mds1 = md_s1(input_frames)
                # fake_mds2_1 = md_s2(fake_mds1)
                # fake_mds2_2 = md_s2(fake_owns2)

                # fake_frames = torch.cat([real_frames, fake_owns1, fake_mds1, fake_mds2_1], dim=4)
                fake_frames = torch.cat([real_frames, fake_owns1, fake_owns_noy], dim=4)

                for batch in range(real_frames.size(0)):
                    frames = fake_frames[batch]
                    frames = frames.unsqueeze(0)
                    image_path = f"{SAVE_DIR}/{used_batch}"
                    os.makedirs(image_path, exist_ok=True)
                    used_batch += 1
                    p_bar.set_description(f"{used_batch}/{(len(loader) *  self.config.batch_size)}")
                    for j in range(self.nframes):
                        a = frames[0][0][j].unsqueeze(0)
                        b = frames[0][1][j].unsqueeze(0)
                        c = frames[0][2][j].unsqueeze(0)
                        A = torch.cat([a, b, c], dim=0)
                    
                        save_image(self.denorm(A.data.cpu()), f'{image_path}/{str(j).zfill(5)}.png')

                    generate_video(
                        inputs=image_path,
                        output_dir=f"{SAVE_DIR}/gifs",
                        filename=f"{used_batch}"
                    )