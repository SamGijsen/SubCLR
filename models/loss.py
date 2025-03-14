import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import os

class ContraWR_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, device: str, batch_size: int, DDP=False,
                 margin=0.2, sigma=2.0, temp=2.0, m=0.9995):
        super(ContraWR_Loss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.DDP = DDP
        self.softmax = torch.nn.Softmax(dim=1)
        self.margin = margin
        self.sigma = sigma
        self.T = temp
        self.m = m

    def forward(self, z_a, z_b):
        if self.DDP:
            z_a = torch.cat(FullGatherLayer.apply(z_a), dim=0)
            z_b = torch.cat(FullGatherLayer.apply(z_b), dim=0)
        z_a = F.normalize(z_a, p=2, dim=1)
        z_b = F.normalize(z_b, p=2, dim=1)

        sim = torch.mm(z_a, z_b.t()) / self.T
        weight = self.softmax(sim)
        neg = torch.mm(weight, z_b)

        l_pos = torch.exp(-torch.sum(torch.pow(z_a - z_b, 2), dim=1) / (2*self.sigma **2))
        l_neg = torch.exp(-torch.sum(torch.pow(z_a - neg, 2), dim=1) / (2*self.sigma **2))

        zero_matrix = torch.zeros(l_pos.shape).to(self.device)
        loss = torch.max(zero_matrix, l_neg - l_pos + self.margin).mean()

        return loss
    
class BYOL_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, device: str, batch_size: int, DDP: bool=False):
        """BYOL Loss."""
        super(BYOL_Loss, self).__init__()
        self.DDP = DDP
        self.batch_size = batch_size
        self.device = device

    def forward(self, z_a, z_b):
        if self.DDP:
            z_a = torch.cat(FullGatherLayer.apply(z_a), dim=0)
            z_b = torch.cat(FullGatherLayer.apply(z_b), dim=0)
        
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)

        loss = -2 * (z_a * z_b).sum()
        loss /= self.batch_size

        return loss
    
class SimCLR_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, device, DDP: bool=False, temp: float=0.05):
        """SimCLR Loss. 
        T: Temperature
        Implementation based on Yang ea. 2023 arXiv:2110.15278"""
        super(SimCLR_Loss, self).__init__()
        self.temp = temp
        self.DDP = DDP
        self.device = device

    def forward(self, z_a, z_b):
        # Embeddings are expected to be [N, D]
        if self.DDP:
            z_a = torch.cat(FullGatherLayer.apply(z_a), dim=0)
            z_b = torch.cat(FullGatherLayer.apply(z_b), dim=0)
        
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)
        N = z_a.shape[0]
        emb_total = torch.cat([z_a, z_b], dim=0)

        # representation similarity matrix
        logits = torch.mm(emb_total, emb_total.t())
        logits /= self.temp 
        logits[torch.arange(2*N), torch.arange(2*N)] = -6e4 
        
        # cross entropy
        labels = torch.LongTensor(torch.cat([torch.arange(N, 2*N), torch.arange(N)])).to(self.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss

class SubCLR_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, device, DDP: bool=False, temp: float=0.05):
        """SubCLR Loss. 
        T: Temperature"""
        super(SubCLR_Loss, self).__init__()
        self.temp = temp
        self.DDP = DDP
        self.device = device

    def forward(self, z, sub_id, online=False):
        # z: Embeddings are expected to be [N, D]
        # sub_id: Subject IDs are expected to be [N, 1]
        if self.DDP:
            z = torch.cat(FullGatherLayer.apply(z), dim=0)
            sub_id = torch.cat(FullGatherLayer.apply(sub_id), dim=0)

        z = F.normalize(z, dim=1)
        N = z.shape[0]

        # representation similarity matrix
        logits = torch.mm(z, z.t())
        logits /= self.temp 

        # find self-similarity and reduce
        logits[torch.arange(N), torch.arange(N)] = -6e4 # FP16-safe

        # Create positive mask based on subject IDs
        pos_mask = (sub_id==sub_id.t()).float()
        pos_mask[torch.arange(N), torch.arange(N)] = 0.
        
        # logsoftmax + nll ('reduction'=mean)
        lsm = torch.log_softmax(logits, dim=1)
        loss = (-lsm*pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
        loss = loss.mean()
        
        if online:
            return loss, logits.cpu().detach().numpy(), sub_id.cpu().detach().numpy()
        else:
            return loss
    
class VICReg_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, device, DDP=False, lam: float=25., mu: float=25., nu: float=1.):
        """
        VICReg loss.
        Lambda: weights the similarity/invariance loss (i.e. MSE between positive pair embeddings)
        Mu: weights the variance loss (i.e. variance of individual embedding dimensions across a batch to avoid collapse)
        Nu: weights the covariance loss (i.e. covariance between embedding dimensions, decorrelating them)
        """
        super(VICReg_Loss, self).__init__()
        self.lam = lam
        self.mu = mu
        self.nu = nu
        self.device = device
        self.DDP = DDP

    def forward(self, z_a, z_b):
        # invariance loss
        sim_loss = F.mse_loss(z_a, z_b)
        
        if self.DDP:
            z_a = torch.cat(FullGatherLayer.apply(z_a), dim=0)
            z_b = torch.cat(FullGatherLayer.apply(z_b), dim=0)
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1-std_z_a)) / 2 + torch.mean(F.relu(1-std_z_b)) / 2

        # covariance loss
        N = z_a.shape[0]
        D = z_a.shape[1]
        cov_a =  (z_a.T @ z_a) / (N-1)
        cov_b = (z_b.T @ z_b) / (N-1)
        cov_loss = off_diagonal(cov_a).pow_(2).sum().div(D
                ) + off_diagonal(cov_b).pow_(2).sum().div(D)
        
        loss = (
            self.lam * sim_loss
            + self.mu * std_loss
            + self.nu * cov_loss
        )

        return loss
    
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
