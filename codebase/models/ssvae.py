import argparse
import numpy as np
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class SSVAE(nn.Module):
    def __init__(self, nn='v1', name='ssvae', gen_weight=1, class_weight=100):
        super().__init__()
        self.name = name
        self.z_dim = 64
        self.y_dim = 10
        self.gen_weight = gen_weight
        self.class_weight = class_weight
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)
        self.cls = nn.Classifier(self.y_dim)

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
        # Compute negative Evidence Lower Bound and its KL_Z, KL_Y and Rec decomposition
        #
        # To assist you in the vectorization of the summation over y, we have
        # the computation of q(y | x) and some tensor tiling code for you.
        #
        # Note that nelbo = kl_z + kl_y + rec
        #
        # Outputs should all be scalar
        ################################################################################
        
        """
        y_logits = self.cls.classify(x)
        y_logprob = F.log_softmax(y_logits, dim=1)
        y_prob = torch.softmax(y_logprob, dim=1) # (batch, y_dim)

        # Duplicate y based on x's batch size. Then duplicate x
        # This enumerates all possible combination of x with labels (0, 1, ..., 9)
        y = np.repeat(np.arange(self.y_dim), x.size(0))
        y = x.new(np.eye(self.y_dim)[y])
        x = ut.duplicate(x, self.y_dim)
        """
        
        # Get original sizez for later
        batch, dim = x.shape

        # Compute the KL divergence term for y
        y_logits = self.cls.classify(x)
        y_logprob = F.log_softmax(y_logits, dim=1)
        y_prob = torch.softmax(y_logprob, dim=1)
        log_prior = torch.log(torch.ones(batch, self.y_dim).to(device) / self.y_dim)
        kl_y = torch.mean(ut.kl_cat(y_prob, y_logprob, log_prior))

        # Duplicate y based on x's batch size, then duplicate x
        # This enumerates all possible combinations of x with labels (0, 1, ..., 9)
        y = np.repeat(np.arange(self.y_dim), batch)
        y_idx = torch.from_numpy(y).type(torch.LongTensor).to(device)
        y = x.new(np.eye(self.y_dim)[y])
        x = ut.duplicate(x, self.y_dim)
        y_prob_dupped = ut.duplicate(y_prob, self.y_dim)

        # Compute the KL divergence term for z
        pm, pv = self.z_prior
        qm, qv = self.enc.encode(x, y)
        probs = torch.sum(y_prob_dupped * y, dim=-1)
        kl_z = torch.sum(probs * ut.kl_normal(qm, qv, pm, pv)) / batch

        # Compute the Reconstruction term
        z = ut.sample_gaussian(qm, qv)
        z_logits = self.dec.decode(z, y)
        rec = - torch.sum(probs * ut.log_bernoulli_with_logits(x, z_logits)) / batch

        # Compute the Negative ELBO
        nelbo = kl_y + kl_z + rec
        
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, kl_y, rec

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls.classify(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, xl, yl):
        if self.gen_weight > 0:
            nelbo, kl_z, kl_y, rec = self.negative_elbo_bound(x)
        else:
            nelbo, kl_z, kl_y, rec = [0] * 4
        ce = self.classification_cross_entropy(xl, yl)
        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_sigmoid_given(self, z, y):
        logits = self.dec.decode(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_x_given(self, z, y):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))
