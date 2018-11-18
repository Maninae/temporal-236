import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################

        # Compute the KL Divergence term
        qm, qv = self.enc.encode(x)
        pm, pv = ut.gaussian_parameters(self.z_pre, dim=1)
        z = ut.sample_gaussian(qm, qv) 
        kl = torch.mean(ut.log_normal(z, qm, qv) - ut.log_normal_mixture(z, pm, pv))

        # Compute the Reconstruction term
        z = ut.sample_gaussian(qm, qv)
        logits = self.dec.decode(z)
        rec = - torch.mean(ut.log_bernoulli_with_logits(x, logits))

        # Compute the Negative ELBO
        nelbo = kl + rec
       
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
      
        # Compute KL and Reconstruction terms
        _, kl, rec = self.negative_elbo_bound(x)

        # Compute relevant terms 
        x = ut.duplicate(x, iw) 
        pm, pv = ut.gaussian_parameters(self.z_pre, dim=1)
        qm, qv = self.enc.encode(x)
        z = ut.sample_gaussian(qm, qv)
        logits = self.dec.decode(z)

        # Compute summands in NIWAE expression `log_args`
        log_pz = ut.log_normal_mixture(z, pm, pv) # log p_{theta}(z)
        log_px_z = ut.log_bernoulli_with_logits(x, logits) # log p_{theta}(x | z)
        log_qz_x = ut.log_normal(z, qm, qv) # log q_{phi}(z | x)
        log_args = log_pz + log_px_z - log_qz_x # log(summand) 

        # Group `iw` samples for each observation together
        log_args = torch.reshape(log_args, (iw, -1))
        log_args = torch.t(log_args)
        
        # Compute NIWAE
        niwae = - torch.mean(ut.log_mean_exp(log_args, dim=-1))

        """
        #### UNVECTORIZED CODE ####
        # Compute NIWAE
        log_args = []
        pm, pv = ut.gaussian_parameters(self.z_pre, dim=1)
        qm, qv = self.enc.encode(x)
        for _ in range(iw):
            z = ut.sample_gaussian(qm, qv)
            logits = self.dec.decode(z)
            log_pz = ut.log_mean_exp(ut.log_normal_mixture(z, pm, pv), dim=-1)
            log_px_z = ut.log_bernoulli_with_logits(x, logits)
            log_qz_x = ut.log_normal(z, qm, qv)
            log_arg = log_pz + log_px_z - log_qz_x
            log_args.append(log_arg)
        log_args = torch.transpose(torch.stack(log_args), 0, 1)
        niwae = - torch.mean(ut.log_mean_exp(log_args, dim=1))

        # Compute KL and Reconstruction terms
        _, kl, rec = self.negative_elbo_bound(x)
        """

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
