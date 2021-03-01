import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import math
import os
from torchvision import transforms


class HeatmapProjection(nn.Module):
    def __init__(self, sigma, landmarks, input_size=256):
        super(HeatmapProjection, self).__init__()
        self.input_size = input_size
        self.var = sigma ** 2
        self.cov = nn.Parameter(torch.eye(2) * sigma
                                ** 2, requires_grad=False)
        yv, xv = torch.meshgrid([torch.arange(0, input_size), torch.arange(0, input_size)])
        self.yv = nn.Parameter(yv.reshape(-1).float(), requires_grad=False)
        self.xv = nn.Parameter(xv.reshape(-1).float(), requires_grad=False)
        self.eye = nn.Parameter(torch.eye(landmarks).float().unsqueeze(0), requires_grad=False)
        self.softmax = LayerWiseSoftmax()

    def forward(self, y):
        """
        Instead of using max activations across each heatmap as landmark coordinates, we weighted average all
        activations across each heatmap. We then re-project landmark coordinates to spatial features with the same
        size as heatmaps by a fixed Gaussian-like function centered at pre-dicted coordinates with a fixed standard
        deviation. As a result, we obtain a new tensor y with the structure prior on content representation.

        """
        # landmarks are the weighted average y
        # activations across each channel
        y_projected = torch.zeros_like(y)
        y = self.softmax(y)

        # self.save_image(y[0, :, :, :])

        # print(y.size())
        # y_0 = y
        y = y.view(y.size(0), y.size(1), -1)
        # print(y[0, :].min(), y[0, :].max())
        mu_y = y * self.yv.reshape(1, 1, -1)
        mu_y = mu_y.sum(dim=-1)  #/ y.sum(dim=-1)
        mu_x = y * self.xv.reshape(1, 1, -1)
        mu_x = mu_x.sum(dim=-1)  #/ y.sum(dim=-1)
        means = torch.cat((mu_y.unsqueeze(-1), mu_x.unsqueeze(-1)), dim=-1)

        # project landmarsk to guassain fuction with fixed standard deviation
        # yv, xv = torch.meshgrid([torch.arange(0, y.size(-2)), torch.arange(0, y.size(-1))])
        # h_act = torch.zeros(y_projected.size(0), y_projected.size(1), device=y.device)
        for batch_id in range(y_projected.size(0)):
            for heatmap_id in range(y_projected.size(1)):
                gdist = MultivariateNormal(means[batch_id, heatmap_id, :], covariance_matrix=self.cov)
                logprobs = gdist.log_prob(torch.cat((self.yv.unsqueeze(0), self.xv.unsqueeze(0)), dim=0).t())
                y_projected[batch_id, heatmap_id, :, :] = torch.exp(logprobs).reshape(y_projected.size(-2), y_projected.size(-1))
                # y_projected[batch_id, heatmap_id, :, :] -= y_projected[batch_id, heatmap_id, :, :].min()
                # y_projected[batch_id, heatmap_id, :, :] /= y_projected[batch_id, heatmap_id, :, :].max()
                # coords = means[batch_id, heatmap_id, :].round().long()
                # h_act[batch_id, heatmap_id] = y_0[batch_id, heatmap_id, coords[0], coords[1]]
                # h_act[batch_id, heatmap_id]
                # max_coords = y_0[batch_id, heatmap_id, :, :].clone()
                # max_coords[max_coords < max_coords.max()] = 0
                # # print(max_coords.size())
                # if batch_id == 0 and heatmap_id == 0:
                #     print(means[batch_id, heatmap_id, :], max_coords.nonzero().float().mean(0))
        # a = 'fail' if y.pow(2).mean() > h_act.mean() else 'pass'
        # print(h_act.mean(), y.mean())
        # print(y_projected.max(), y_projected.min())
        # y_out = y_projected.sum(dim=1)
        # self.save_image(y_out.unsqueeze(1), normalize=True)
        # self.save_image(y_out.unsqueeze(1), normalize=True, output='landmarks')
        return y_projected, self.prior_loss(means, y)

    def prior_loss(self, h_ij, y, lam_1=1, lam_2=1):
        var_h = self.conc_loss(h_ij, y)
        h_ij /= self.input_size
        distance = batch_pairwise_distances(h_ij, h_ij)
        var = self.var / (self.input_size ** 2)
        sep_loss = torch.exp(-distance / (2 * var))
        sep_loss = 0.5 * sep_loss * (1 - self.eye.expand_as(sep_loss))

        prior_loss = lam_1 * sep_loss.view(sep_loss.size(0), -1).sum(-1) + lam_2 * var_h.mean(-1)

        return prior_loss.mean()

    def conc_loss(self, means, y):
        # raise NotImplementedError
        # y spatial variance
        sig_y = self.yv.reshape(1, 1, -1).expand_as(y) - means[:, :, 0].unsqueeze(-1)
        sig_y /= self.input_size
        sig_y = y * (sig_y.pow(2))
        sig_y = sig_y.sum(-1)
        # print(sig_y.size())

        # x spatial variance
        sig_x = self.xv.reshape(1, 1, -1).expand_as(y) - means[:, :, 1].unsqueeze(-1)
        sig_x /= self.input_size
        sig_x = y * (sig_x.pow(2))
        sig_x = sig_x.sum(-1)

        # print(sig_y)
        # conc_loss = torch.exp(sig_y + sig_x)
        conc_loss = 0.5 * (sig_y + sig_x)
        return conc_loss.mean(-1)

    @staticmethod
    def save_image(images, normalize=True, output=None):
        for i in range(4):
            if i < images.size(0):
                image = images[i, :, :].cpu()
                if output is None:
                    out_path = 'heatmaps/example-{}.jpg'.format(i)
                else:
                    out_path = 'heatmaps/{}.jpg'.format(output)
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                if normalize:
                    image -= image.min()
                    image /= image.max()
                transf = transforms.ToPILImage()
                image = transf(image).convert('RGB')
                image.save(out_path)


def batch_pairwise_distances(x, y):
    """
    Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    Input: x is a bxNxd matrix y is an optional bxMxd matirx
    Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
    i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
    """
    x_norm = (x**2).sum(2).view(x.shape[0], x.shape[1],1)
    y_t = y.permute(0, 2, 1).contiguous()
    y_norm = (y**2).sum(2).view(y.shape[0], 1, y.shape[1])
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    dist[dist != dist] = 0  # replace nan values with 0
    return torch.clamp(dist, 0.0, np.inf)


class LayerWiseSoftmax(nn .Module):
    def __init__(self):
        super(LayerWiseSoftmax, self).__init__()

    def forward(self, x):
        x = F.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)

        # # check sum over reach channel == 1
        # assert x[0, 0, :, :].sum()
        # assert x[1, 2, :, :].sum()
        # assert x[2, 3, :, :].sum()
        # assert x[4, 6, :, :].sum()
        # assert x[3, 8, :, :].sum()
        return x
