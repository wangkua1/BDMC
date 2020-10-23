from __future__ import print_function

import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import grad as torchgrad
from BDMC import hmc
from BDMC import utils
#
import matplotlib.pylab as plt
import torchvision.utils as vutils
import os

def ais_trajectory(model,
                   model_latent_dim,
                   model_decode_vector,
                   loader,
                   forward=True,
                   schedule=np.linspace(0., 1., 500),
                   n_sample=100,
                   log_likelihood_fn=utils.log_bernoulli,
                   log_prior_fn=None,
                   prior_sample_fn=None,
                   save_dir=None):
  """Compute annealed importance sampling trajectories for a batch of data. 
  Could be used for *both* forward and reverse chain in BDMC.

  Args:
    model (vae.VAE): VAE model
    loader (iterator): iterator that returns pairs, with first component
      being `x`, second would be `z` or label (will not be used)
    forward (boolean): indicate forward/backward chain
    schedule (list or 1D np.ndarray): temperature schedule, i.e. `p(z)p(x|z)^t`
    n_sample (int): number of importance samples

  Returns:
      A list where each element is a torch.autograd.Variable that contains the 
      log importance weights for a single batch of data
  """
  if log_prior_fn is None: 
    log_prior_fn = lambda z: utils.log_normal(z, torch.zeros_like(z), torch.zeros_like(z))
  if prior_sample_fn is None:
    prior_sample_fn = lambda B: torch.randn(B, model_latent_dim).cuda()
  
  def log_f_i(z, data, t):
    """Unnormalized density for intermediate distribution `f_i`:
        f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
    =>  log f_i = log p(z) + t * log p(x|z)
    """
    log_prior = log_prior_fn(z)
    log_likelihood = log_likelihood_fn(model_decode_vector(z), data)

    return log_prior + log_likelihood.mul_(t)

  logws = []
  for i, (batch, post_z) in enumerate(loader):
    B = batch.size(0) * n_sample
    batch = batch.cuda()
    batch = utils.safe_repeat(batch, n_sample)

    with torch.no_grad():
      epsilon = torch.ones(B).cuda().mul_(0.01)
      accept_hist = torch.zeros(B).cuda()
      logw = torch.zeros(B).cuda()

    # initial sample of z
    if forward:
      current_z = prior_sample_fn(B)
    else:
      current_z = utils.safe_repeat(post_z, n_sample).cuda()
    # current_z = current_z.requires_grad_()
    pbar = tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1))
    for j, (t0, t1) in pbar:

      current_z = current_z.detach()
      current_z.requires_grad_()
      # update log importance weight
      log_int_1 = log_f_i(current_z, batch, t0).detach()
      log_int_2 = log_f_i(current_z, batch, t1).detach()
      logw += log_int_2 - log_int_1

      # resample velocity
      current_v = torch.randn(current_z.size()).cuda()

      def U(z):
        return -log_f_i(z, batch, t1).detach()

      def grad_U(z):
        # grad w.r.t. outputs; mandatory in this case
        grad_outputs = torch.ones(B).cuda()
        # torch.autograd.grad default returns volatile
        # grad = torchgrad(U(z), z, grad_outputs=grad_outputs)[0]
        grad = torchgrad(-log_f_i(z, batch, t1), z, grad_outputs=grad_outputs,retain_graph=False, create_graph=False)[0]
        # clip by norm
        max_ = B * model_latent_dim * 100.
        grad = torch.clamp(grad, -max_, max_)
        # grad.requires_grad_()
        return grad.detach()

      def normalized_kinetic(v):
        zeros = torch.zeros(B, model_latent_dim).cuda()
        return -utils.log_normal(v, zeros, zeros).detach()

      z, v = hmc.hmc_trajectory(current_z, current_v, U, grad_U, epsilon)
      # import ipdb; ipdb.set_trace()
      current_z, epsilon, accept_hist = hmc.accept_reject(
          current_z, current_v,
          z, v,
          epsilon,
          accept_hist, j,
          U, K=normalized_kinetic)
      tmp = utils.log_mean_exp(logw.view(n_sample, -1).transpose(0, 1)).mean().item()
      pbar.set_postfix_str(s=f'AIS: {tmp:.2f}', refresh=True)
      if j % 10 == 1: 
        with torch.no_grad():
          x = model_decode_vector(z)
        h = w = int(np.sqrt(x.size(1)))
        # Plot
        def _plot(ims):
          c, h, w = ims.shape[-3:]
          grid = vutils.make_grid(ims.reshape(-1, c,h,w), nrow=8, padding=2, normalize=True) 
          plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
          plt.tight_layout()
          plt.grid()
          plt.xticks([])
          plt.yticks([])
        f, axs = plt.subplots(1, 2, figsize=(8,5))
        plt.subplot(axs[0])
        _plot(x.view(x.size(0),1,h, w)[:64].cpu())
        plt.subplot(axs[1])
        _plot(batch.view(x.size(0),1,h, w)[:64].cpu())
        plt.savefig(os.path.join(save_dir,f'{j}.jpeg'), bbox_inches='tight', pad_inches=0, format='jpeg')
        
        


    logw = utils.log_mean_exp(logw.view(n_sample, -1).transpose(0, 1))
    if not forward:
      logw = -logw
    logws.append(logw.data.detach().cpu().numpy())
    print('Last batch stats %.4f' % (logw.mean().cpu().data.numpy()))

    # model.zero_grad()
    # del log_int_1
    # del log_int_2
    # del logw
  return logws
