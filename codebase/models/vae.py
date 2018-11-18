import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

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
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################

        # Compute KL divergence term
        qm, qv = self.enc.encode(x)
        pm, pv = self.z_prior
        kl = torch.mean(ut.kl_normal(qm, qv, pm, pv))

        # Compute reconstruction term
        z = ut.sample_gaussian(qm, qv)
        logits = self.dec.decode(z)
        rec = - torch.mean(ut.log_bernoulli_with_logits(x, logits))

        # Compute negative ELBO
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

        # TODO: Vectorize this code

        # Compute KL and Reconstruction terms
        _, kl, rec = self.negative_elbo_bound(x)

        # Compute relevant terms for NIWAE calculation
        x = ut.duplicate(x, iw)
        pm, pv = self.z_prior_m.data, self.z_prior_v.data
        qm, qv = self.enc.encode(x)
        z = ut.sample_gaussian(qm, qv)
        logits = self.dec.decode(z)

        # Compute NIWAE
        log_pz = ut.log_normal(z, pm, pv)
        log_px_z = ut.log_bernoulli_with_logits(x, logits)
        log_qz_x = ut.log_normal(z, qm, qv)
        log_args = log_pz + log_px_z - log_qz_x

        # Group `iw` samples for each observation together
        log_args = torch.reshape(log_args, (iw, -1))
        log_args = torch.t(log_args)

        # Compute NIWAE
        niwae = - torch.mean(ut.log_mean_exp(log_args, dim=1))

        """
        #### UNVECTORIZED CODE ####
        # Compute NIWAE
        log_args = []
        pm, pv = self.z_prior_m.data, self.z_prior_v.data
        qm, qv = self.enc.encode(x)
        for _ in range(iw):
            z = ut.sample_gaussian(qm, qv)
            logits = self.dec.decode(z)
            log_pz = ut.log_normal(z, pm, pv)
            log_px_z = ut.log_bernoulli_with_logits(x, logits)
            log_qz_x = ut.log_normal(z, qm, qv)
            log_arg = log_pz + log_px_z - log_qz_x
            log_args.append(log_arg)
        log_args = torch.transpose(torch.stack(log_args), 0, 1)
        niwae = - torch.mean(ut.log_mean_exp(log_args, dim=1))

        # Compute KL divergence and Reconstruction terms
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
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
