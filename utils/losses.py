#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import math

# configuration module
# -----
import config

# utilities
# -----
import torch.nn.functional as F
import torch.nn as nn
import torch


# custom classes
# -----



class RELIC(nn.Module):
    """
        RELIC loss which minimizes similarities the same between the anchor and different views of other samples,
            i.e. x and its pair x_pair
    """

    def __init__(self, args, sim_func, **kwargs):
        """Initialize the RELIC_TT_Loss class"""
        super(RELIC, self).__init__()
        self.sim_func = sim_func
        self.args = args
        assert self.args.num_devices == 1


    def forward(self, x, x_pair):
        """
        params:
            x: representation tensor (Tensor)
            x_pair: tensor of the same size as x which should be the pair of x (Tensor)
        return:
            loss: the loss of RELIC-TT (Tensor)
        """
        sim = self.sim_func(x, x_pair)
        loss = F.kl_div(sim.softmax(-1).log(), sim.T.softmax(-1), reduction='batchmean')
        return loss


class BYOL(nn.Module):
    """
        BYOL loss that maximizes cosine similarity between the online projection (x) and the target projection(x_pair)
    """

    @classmethod
    def get_byol_output(cls, proj_target_output, pred_output):
        mid_size = pred_output.shape[0] // 2

        x_mix = torch.cat((pred_output[:mid_size], proj_target_output[:mid_size]), dim=0)
        y_mix = torch.cat((proj_target_output[mid_size:], pred_output[mid_size:]), dim=0)
        return x_mix, y_mix

    @classmethod
    def get_args(cls, parser):
        return parser

    def __init__(self, args, sim_func, fabric, **kwargs):
        """Initialize the SimCLR_TT_Loss class"""
        super(BYOL, self).__init__()
        self.args = args
        self.fabric = fabric

    def forward(self, x, x_target):
        """
        params:
            x: representation tensor (Tensor)
            x_pair: tensor of the same size as x which should be the pair of x (Tensor)
        return:
            loss: the loss of BYOL-TT (Tensor)
        """
        loss = 2-2*F.cosine_similarity(x, x_target, dim=1).mean()
        return loss




class SimCLR(nn.Module):
    def __init__(self, args, sim_func, fabric, batch_size=None, temperature=None, **kwargs):
        """Initialize the SimCLR_TT_Loss class"""
        super(SimCLR, self).__init__()

        self.args = args
        self.fabric = fabric
        self.batch_size = args.batch_size if batch_size is None else batch_size
        self.temperature = args.temperature if temperature is None else temperature

        # self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.sim_func = sim_func


    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--n_negative', default=None, type=int)
        return parser

    def forward(self, x, x_pair, labels=None, negatives=True):
        """
        Given a positive pair, we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        to control for negative samples we just cut off losses
        """


        N = 2 * self.batch_size * self.args.num_devices

        z_local = torch.cat((x, x_pair), dim=0)
        z_all= self.fabric.all_gather(z_local, sync_grads=True).view(-1, z_local.shape[1])
        z_rank_list = z_all.split(self.args.batch_size)
        z_pos = torch.cat([z_rank_list[zi*2] for zi in range(len(z_rank_list)//2)], dim=0)
        z_pos2 = torch.cat([z_rank_list[1+zi*2] for zi in range(len(z_rank_list)//2)], dim=0)

        z = torch.cat((z_pos, z_pos2), dim=0)
        sim = self.sim_func(z, z) / self.temperature
        # get the entries corresponding to the positive pairs
        sim_i_j = torch.diag(sim, self.batch_size*self.args.num_devices)
        sim_j_i = torch.diag(sim, -self.batch_size*self.args.num_devices)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # Much faster than masking !
        # mask = (torch.ones_like(sim) - torch.eye(2 * self.args.batch_size, device=sim.device)).bool()
        # [2*B, 2*B-1]
        # logits = sim.masked_select(mask).view(2 * self.args.batch_size, -1)

        logits = sim.flatten()[1:].view(N-1, N+1)[:,:-1].reshape(N, N-1)
        negative_loss = torch.logsumexp(logits, dim=1,keepdim=True)

        if not negatives:
            return -positive_samples.mean()
        # negative_loss = (negative_loss.squeeze() ** self.q_rince)/self.q_rince
        # scale_factor = self.args.num_devices if self.args.rescale else 1
        # print(-positive_samples.mean().item(), negative_loss.mean().item(), (-positive_samples.mean() + negative_loss.mean()).mean().item() )
        # print(-positive_samples.mean() + negative_loss.mean())
        # print(-positive_samples.mean(), negative_loss.mean(), sim.mean(), sim.shape)

        return (-positive_samples.mean() + negative_loss.mean())
        # return negative_loss.mean()
        # return -positive_samples.mean()


class Rince(nn.Module):
    def __init__(self, args, sim_func, **kwargs):
        """https://openaccess.thecvf.com/content/CVPR2022/papers/Chuang_Robust_Contrastive_Learning_Against_Noisy_Views_CVPR_2022_paper.pdf"""
        super().__init__()
        self.args=args
        # Standard hyper-parameters of the algorithm
        self.q_rince = 0.5
        self.lambda_rince = 0.01
        self.temperature = self.args.temperature
        self.sim_func = sim_func
        self.mask = torch.ones(2 * args.batch_size, 2 * args.batch_size, dtype=torch.bool)
        self.mask = self.mask.fill_diagonal_(0)
        for i in range(args.batch_size):
            self.mask[i, args.batch_size + i] = 0
            self.mask[args.batch_size + i, i] = 0
        assert self.args.num_devices == 1


    @classmethod
    def get_args(cls, parser):
        return parser

    def forward(self, x, x_pair):
        z = torch.cat((x, x_pair), dim=0)
        sim = self.sim_func(z, z) / self.temperature
        sim_i_j = torch.diag(sim, sim.shape[0]//2)
        sim_j_i = torch.diag(sim, -sim.shape[0]//2)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(sim.shape[0], 1)
        positive_loss = torch.exp(self.q_rince * positive_samples) / self.q_rince

        negative_samples = sim[self.mask].reshape(sim.shape[0], -1)
        negative_loss = self.lambda_rince * (torch.exp(positive_samples) + torch.sum(torch.exp(negative_samples), dim=1,keepdim=True))
        negative_loss = (negative_loss.squeeze() ** self.q_rince)/self.q_rince

        return (-positive_loss + negative_loss).mean()


class MultiPositiveSimCLR(nn.Module):
    def __init__(self, args, sim_func, sim_func_simple, n_positives, temperature=None, **kwargs):
        """Initialize the MultiPosSimCLR_TT_Loss class"""
        super(MultiPositiveSimCLR, self).__init__()
        assert args.similarity == "cosine", "only handle cosine similarity for now"
        self.args = args
        self.batch_size = args.batch_size * (1+n_positives)

        self.temperature = args.temperature if temperature is None else temperature
        self.n_positives = n_positives

        self.mask = self.mask_correlated_samples(self.batch_size)
        self.sim_func = sim_func
        self.sim_func_simple = sim_func_simple
        assert self.args.num_devices == 1

    def mask_correlated_samples(self, batch_size):
        """
        mask_correlated_samples takes the int batch_size
        and returns an np.array of size [2*batchsize, 2*batchsize]
        which masks the entries that are the same image or
        the corresponding pogetattr(getattr(sitive contrast
        """
        mask = torch.ones(batch_size, batch_size, dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        return mask

    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--multi_pos_weight', default=1, type=float)
        return parser

    def forward(self, x, x_pair, **kwargs):

        z = torch.cat((x, x_pair), dim=0)
        # if self.args.multiscale_version == 1:
        sim = self.sim_func(z, z) / self.temperature
        loss_pos = 0
        # for p in x.split(2*self.args.batch_size):
        for p in x.split(self.args.batch_size):
            loss_pos += (self.sim_func_simple(p, x_pair)/self.temperature).mean(dim=0)
        loss_pos = self.args.multi_pos_weight * loss_pos / self.n_positives
        negative_samples = sim[self.mask].reshape(self.batch_size, -1)

        # elif self.args.multiscale_version == 2:
        #     sim = self.sim_func(x, x) / self.temperature
        #     loss_pos = 0
        #     sim_pos= []
        #     for p in x.split(2*self.batch_size):
        #         pos = self.sim_func_simple(p, x_pair)/self.temperature
        #         sim_pos.append(pos)
        #         loss_pos += pos.mean(dim=0)
        #     loss_pos = self.args.multi_pos_weight * loss_pos / self.n_positives
        #     negative_samples = sim[self.mask].reshape(2 * self.batch_size * self.n_positives, -1)
        #     negative_samples = torch.cat((torch.cat(sim_pos, dim=0).view(-1, 1), negative_samples), dim=1)
        # p = x.view(2*self.batch_size, self.n_positives, self.args.feature_dim)
        # sim_pos = F.cosine_similarity(p, x_pair.unsqueeze(1),dim=2)/self.temperature
        # loss_pos = torch.logsumexp(sim_pos, dim=1).mean(dim=0)


        # we take all of the negative samples

        if self.args.n_negative:
            negative_samples = torch.take_along_dim(
                negative_samples, torch.rand(*negative_samples.shape, device=self.args.device).argsort(dim=1), dim=1)
            negative_samples = negative_samples[:, :self.args.n_negative]

        loss_neg = torch.logsumexp(negative_samples, dim=1).mean(dim=0)

        return loss_neg - loss_pos




class BatchWiseSimCLR(nn.Module):
    def __init__(self, arguments, *args, temperature=None, mask_size=512, **kwargs):
        """Initialize the SimCLR_TT_Loss class"""
        super(BatchWiseSimCLR, self).__init__()

        self.args = arguments
        self.batch_size = self.args.batch_size
        self.temperature = self.args.temperature if temperature is None else temperature
        self.mask = self.mask_correlated_samples(self.batch_size, mask_size)
        self.mask_size = mask_size
        assert self.args.num_devices == 1


    def mask_correlated_samples(self, batch_size, mask_size):
        """
        mask_correlated_samples takes the int batch_size
        and returns an np.array of size [2*batchsize, 2*batchsize]
        which masks the entries that are the same image or
        the corresponding pogetattr(getattr(sitive contrast
        """
        mask = torch.ones((2 * batch_size, mask_size, mask_size), dtype=torch.bool)
        for i in range(mask.shape[0]):
            mask[i].fill_diagonal_(0)


        return mask

    @classmethod
    def get_args(cls, parser):
        return parser

    def forward(self, x, x_pair,**kwargs):
        """
        Given a positive pair, we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        to control for negative samples we just cut off losses
        """
        n_positives = x.shape[0] / x_pair.shape[0]
        for z in x.split(x_pair.shape[0]):
            loss_pos = (F.cosine_similarity(z, x_pair, dim=1)/self.temperature)
        loss_pos /= n_positives
        loss_pos = loss_pos.mean()



        x_reshape = torch.cat([x_i.unsqueeze(1) for x_i in x.split(self.args.batch_size*2)],dim=1)
        x_pair_reshape = torch.cat([x_i.unsqueeze(1) for x_i in x_pair.split(self.args.batch_size*2)],dim=1)
        z = torch.cat((x_reshape, x_pair_reshape), dim=1)
        sim = F.cosine_similarity(z.unsqueeze(2), z.unsqueeze(1), dim=3)/self.temperature
        negative_samples = sim[self.mask].reshape(2*self.args.batch_size, self.mask_size, self.mask_size-1)
        loss_neg = torch.logsumexp(negative_samples, dim=2).mean()
        return loss_neg - loss_pos

class VicReg(nn.Module):
    def __init__(self, argss, sim_func, fabric, *args, lambda_vicreg=None, mu_vicreg=None, v_vicreg=None, **kwargs):
        super().__init__()
        self.fabric = fabric
        self.args = argss
        self.lambda_vicreg = self.args.lambda_vicreg if lambda_vicreg is None else lambda_vicreg
        self.mu_vicreg = self.args.mu_vicreg if mu_vicreg is None else mu_vicreg
        self.v_vicreg = self.args.v_vicreg if v_vicreg is None else v_vicreg

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    @classmethod
    def get_args(cls, parser):
        return parser

    def forward(self, x_local, x_pair_local, *args, **kwargs):
        x = self.fabric.all_gather(x_local, sync_grads=True).view(-1, x_local.shape[1])
        x_pair = self.fabric.all_gather(x_pair_local, sync_grads=True).view(-1, x_pair_local.shape[1])


        repr_loss = F.mse_loss(x, x_pair)


        x = x - x.mean(dim=0)
        x_pair = x_pair - x_pair.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(x_pair.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (x_pair.T @ x_pair) / (x.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(x.shape[1]) + self.off_diagonal(cov_y).pow_(
            2).sum().div(x.shape[1])
        return self.v_vicreg*cov_loss + self.mu_vicreg*std_loss + self.lambda_vicreg*repr_loss


class InfoNCELoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.targets = torch.arange(0, batch_size, dtype=torch.long).to(device)
        assert self.args.num_devices == 1

    @classmethod
    def get_args(cls, parser):
        return parser

    def forward(self, z1, z2):
        logits = F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T / self.temperature
        return F.cross_entropy(logits, self.targets) / 2 + F.cross_entropy(logits.T, self.targets) / 2

# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
