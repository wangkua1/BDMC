from __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable
from torch.distributions import Bernoulli


def simulate_data(model_latent_dim, model_decode, batch_size=10, n_batch=1,act_x=torch.sigmoid):
  """Simulate data from the VAE model. Sample from the 
  joint distribution p(z)p(x|z). This is equivalent to
  sampling from p(x)p(z|x), i.e. z is from the posterior.

  Bidirectional Monte Carlo only works on simulated data,
  where we could obtain exact posterior samples.

  Args:
      model: VAE model for simulation
      batch_size: batch size for simulated data
      n_batch: number of batches

  Returns:
      iterator that loops over batches of torch Tensor pair x, z
  """

  # shorter aliases

  batches = []
  for i in range(n_batch):
    # assume prior for VAE is unit Gaussian
    z = torch.randn(batch_size, model_latent_dim).cuda()
    x_logits = model_decode(z)
    if isinstance(x_logits, tuple):
      x_logits = x_logits[0]
    x_bernoulli_dist = Bernoulli(probs=act_x(x_logits))
    x = x_bernoulli_dist.sample().data

    paired_batch = (x, z)
    batches.append(paired_batch)

  return iter(batches)
