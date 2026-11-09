import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

from tqdm import tqdm

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.Module):
    """用于训练的高斯扩散模型"""
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels, fibre_density):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.smooth_l1_loss(self.model(x_t, t, labels, fibre_density), noise, reduction='sum', beta=1.0)
        return loss

class GaussianDiffusion_ddim(nn.Module):
    """DDIM采样器"""
    def __init__(self, model, beta_1, beta_T, image_size_x, image_size_y, batch_size, channels, device,
                 ddim_timesteps=50, ddim_discr_method='uniform', ddim_eta=1.0, clip_denoised=True,
                 timesteps=1000, beta_schedule='linear', U=3):
        super(GaussianDiffusion_ddim, self).__init__()
        self.model = model
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.batch_size = batch_size
        self.channels = channels
        self.ddim_timesteps = ddim_timesteps
        self.ddim_discr_method = ddim_discr_method
        self.ddim_eta = ddim_eta
        self.clip_denoised = clip_denoised
        self.timesteps = timesteps
        self.beta_schdule = beta_schedule
        self.U = U

        self.device = device
        betas = torch.linspace(beta_1, beta_T, timesteps, device=device)
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def forward(self, x_T, label, fibre_density):
        # make ddim timestep sequence
        if self.ddim_discr_method == 'uniform':
            c = self.timesteps // self.ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif self.ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.timesteps * .8), self.ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{self.ddim_discr_method}"')
        
        ddim_timestep_seq = ddim_timestep_seq + 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = self.device
        sample_img = x_T
        
        for i in tqdm(reversed(range(0, self.ddim_timesteps)), desc='sampling loop time step', total=self.ddim_timesteps):
            t = torch.full((self.batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((self.batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            xishu = self.cal_alpha_cump(prev_t, t)
            
            alpha_cumprod_t = extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, sample_img.shape)
            
            for u in range(1, self.U + 1):
                pred_noise = self.model(sample_img, t, label, fibre_density)
                pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
                
                if self.clip_denoised:
                    pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

                sigmas_t = self.ddim_eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

                pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise
                x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)
                
                if u < self.U and i >= 1:
                    sample_img = xishu[0] * x_prev + xishu[1] * torch.randn_like(x_prev)
                else:
                    sample_img = x_prev
                    
        return sample_img

    def cal_alpha_cump(self, t_pred, t):
        """计算由前一子集的x_t-1到x_t的alpha的累乘"""
        a = 1
        for ii in range(t_pred + 1, t + 1):
            a = a * self.alphas[ii]
        return torch.sqrt(a), torch.sqrt(1 - a)
