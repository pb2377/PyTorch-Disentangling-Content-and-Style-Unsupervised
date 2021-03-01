import torch
import torch.nn as nn
import hourglasses
import modules.vae as V
import modules.landmark_projection as lp
import torch.nn.functional as F
import vgg
from torchvision.transforms import Normalize
from time import process_time


# # # # # # # # # # DISENTANGLING CONTENT AND STYLE VIA UNSUPERVISED GEOMETRY DISTILLATION # # # # # # # # # #
class GeomVAE(nn.Module):
    """VAE of DISENTANGLING CONTENT AND STYLE VIA UNSUPERVISED GEOMETRY DISTILLATION"""
    def __init__(self, in_channels=3, landmarks=30, sigma=2):
        super(GeomVAE, self).__init__()
        num_channels = 16  # from hourglass paper

        # Structure Branch
        self.structure_branch = hourglasses.StackedHourGlass(in_channels=in_channels, nChannels=num_channels, nStack=1,
                                                             nModules=2, numReductions=4, nJoints=landmarks)
        # self.structure_branch = hg2.Hourglass()
        self.project_y = lp.HeatmapProjection(sigma=sigma, landmarks=landmarks)

        self.reduce_y = False  # reduce y to 1 channel after extraction
        if self.reduce_y:
            landmarks = 1

        self.y_norm = nn.InstanceNorm2d(landmarks)
        self.encode_structure = EncoderVAE(in_channels=landmarks, need_skips=True)

        # Style Branch
        self.encode_style = EncoderVAE(in_channels=in_channels+landmarks, need_skips=False)

        # Decoder
        self.skip_mode = 'cat'
        # self.skip_mode = 'add'
        decoder_in = 128
        self.decoder = DecoderVAE(in_channels=decoder_in, out_channels=in_channels, skip_mode=self.skip_mode)

        # batch_norm = True
        # self.vgg_extractor = ExtractorVGG(features=vgg.make_layers(vgg.cfgs['E'], batch_norm=batch_norm),
        #                                   arch='vgg19', batch_norm=batch_norm).eval()
        self.L1_loss = nn.L1Loss()

    def forward(self, x):
        # # Structure y through hourglass
        t = [process_time()]
        y = self.structure_branch(x)
        y = y[0]

        t.append(process_time())
        y = F.interpolate(y, size=(256, 256), mode='nearest')
        # y = F.interpolate(y, size=(256, 256), mode='bilinear', align_corners=True)
        y, prior_loss = self.project_y(y)

        # y = y.detach()
        t.append(process_time())
        if self.reduce_y:
            y = y.sum(1, keepdim=True)
            # y, _ = y.max(dim=1)
            # y = y.unsqueeze(1)

        y = self.y_norm(y)
        # y = self.y_norm(y)
        # print(y.max(), y.min())
        z_structure, skips = self.encode_structure(y)

        # # Style branch
        # # x_y = torch.cat((x, y), dim=1)  # concatenate x with structure, y, to encode style p(z|x, y)
        t.append(process_time())
        z_style = self.encode_style(torch.cat((x, y), dim=1))

        # # By modeling the two distributions as Gaussian with identity covariances,
        # # the KL Loss is simply equal to the Euclidean distance  between their means
        t.append(process_time())
        kl_loss = 0.5 * torch.pow(z_style.squeeze() - z_structure.squeeze(), 2).sum(-1)
        kl_loss = kl_loss.mean()

        t.append(process_time())
        z_style = self.reparameterize(z_style)
        if self.skip_mode == 'cat':
            z = torch.cat((z_style, z_structure), dim=1)
        else:
            z = z_style + z_structure  # fuse features, essentially the first skip layers
        x_out = self.decoder(z, skips=skips)

        t.append(process_time())
        rc_loss = self.reconstruction_loss(x, x_out)
        t.append(process_time())

        # delta_t = []
        # for i in range(1, len(t)):
        #     delta_t.append(t[i] - t[i-1])
        # print(delta_t)
        return x_out, y, (rc_loss, prior_loss, kl_loss)

    @staticmethod
    def reparameterize(mu, logvar=None):
        if logvar is None:
            # in the paper, they just estimate mu, and use an identify matrix as sigma
            logvar = torch.ones_like(mu)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reconstruction_loss(self, x_in, x_out):
        """
        Calculate the reconstruction loss of the whole model using combined losses for L1 loss between the image apir,
        and loss between the image pairs' features from the l^th layer of VGG-19.
        **Authors note it would also be possible to add an adversarial loss too

        :param x_in: Original Image
        :param x_out: Reconstructed Image
        :param lam: Weighting factor for feature map losses
        :return:
        """
        # self.vgg_extractor.eval()
        # calculate L1(?) loss between x_in, x_out
        x_loss = self.L1_loss(x_in, x_out)
        return x_loss


class EncoderVAE(nn.Module):
    def __init__(self, in_channels, need_skips=False):
        super(EncoderVAE, self).__init__()
        self.need_skips = need_skips
        # self.layer_0 = nn.Sequential(nn.InstanceNorm2d(in_channels),
        #                              nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=True))
        self.layer_0 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=True)
        main_layers = []
        # Arch first layer is 64-128, rest are 128-128 channels
        arch = [(64, 128, 4, 2)] + [(128, 128, 4, 2)] * 5 + [(128, 128, 4, 2, 1, True)]
        # final layer maps to a vector so can't instant norm
        for layer in arch:
            main_layers.append(V.LeakyBlock(*layer))
        self.main_layers = nn.ModuleList(main_layers)
        self.layer_mu = nn.Conv2d(128, 128, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.layer_0(x)
        skips = [x.clone()]
        for i, layer in enumerate(self.main_layers):
            x = layer(x)
            if self.need_skips:
                skips.append(x.clone())
            assert not torch.equal(x[0, :, :, :], x[1, :, :, :])

        x = self.layer_mu(x)
        if self.need_skips:
            return x, skips
        else:
            return x


class DecoderVAE(nn.Module):
    """
    Using upsample->skip->conv instead of skip->deconv
    """
    def __init__(self, in_channels, out_channels, skip_mode='cat'):
        super(DecoderVAE, self).__init__()
        decoder_channels = 128
        if skip_mode == 'cat':
            in_channels *= 2
            arch = [(decoder_channels*2, decoder_channels, 4, 2)] * 6 + [(decoder_channels*2, 64, 4, 2)]
            final_channels = 128
        elif skip_mode == 'add':
            arch = [(decoder_channels, decoder_channels, 4, 2)] * 6 + [(decoder_channels, 64, 4, 2)]
            final_channels = 64
        else:
            print('Invalid skip_mode')
            raise NotImplementedError
        # opposite to encoder so layer_mu is first, then layer_9 at the end
        self.conv_0 = nn.Conv2d(in_channels, decoder_channels, kernel_size=1, stride=1, padding=0)
        main_layers = []
        for layer in arch:
            main_layers.append(V.DeconvBlock(*layer))
        self.main_layers = nn.ModuleList(main_layers)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.layer_end = nn.ConvTranspose2d(final_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.skip_mode = skip_mode

    def forward(self, x, skips=None):
        x = self.conv_0(x)
        skips = skips[::-1]
        for layer_id, layer in enumerate(self.main_layers):
            if skips is not None:
                x = self.skip_layer(x, skips[layer_id])
            x = layer(x)
        x = self.skip_layer(x, skips[-1])
        x = self.tanh(self.layer_end(self.relu(x)))
        return x

    def skip_layer(self, x, skip_x):
        if self.skip_mode == 'cat':
            return torch.cat((x, skip_x), dim=1)
        else:
            return x + skip_x


class ExtractorVGG(vgg.VGG):
    def __init__(self, features, arch, batch_norm):
        super(ExtractorVGG, self).__init__(features)
        # check for batch norm
        if batch_norm:
            arch = 'vgg19_bn'
        self.load_weights(arch)
        del self.classifier

        # # extract features before every Maxpool layer
        # if not batch_norm:
        #     self.extract_ids = [3, 8, 17, 26, 35]
        # else:
        #     self.extract_ids = [5, 12, 25, 38, 51]

        # extract features after every Maxpool layer
        if not batch_norm:
            self.extract_ids = [4, 9, 18, 27, 36]
        else:
            self.extract_ids = [6, 13, 26, 39, 52]

        self.loss = nn.L1Loss()

    def _forward(self, x):
        # normalize images
        # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.normalize(x)

        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.extract_ids:
                outputs.append(x)

        return tuple(outputs)

    def load_weights(self, arch, progress=True):
        state_dict = vgg.load_state_dict_from_url(vgg.model_urls[arch],
                                                  progress=progress)
        state_dict = {k: v for k, v in state_dict.items() if k in self.state_dict()}
        self.load_state_dict(state_dict)

    def forward(self, x_in, x_out, lam=1.):
        # def reconstruction_loss(self, x_in, x_out, lam=1.):
        """
        Calculate the reconstruction loss of the whole model using combined losses for L1 loss between the image apir,
        and loss between the image pairs' features from the l^th layer of VGG-19.
        **Authors note it would also be possible to add an adversarial loss too

        :param x_in: Original Image
        :param x_out: Reconstructed Image
        :param lam: Weighting factor for feature map losses
        :return:
        """
        # calculate L1(?) losses between l-th vgg features f_in, f_out for all l
        x_loss = 0.

        # concat inputs
        batch_size = x_in.size(0)
        x = torch.cat((x_in, x_out), dim=0)
        fmaps = self._forward(x)
        for layer_id in range(len(fmaps)):
            x_loss += lam * self.loss(fmaps[layer_id][:batch_size, :, :, :], fmaps[layer_id][batch_size:, :, :, :])
        return x_loss

    @staticmethod
    def normalize(x):
        # Normalize with imagenet mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x_new = x.clone()
        for channel in range(3):
            x_new[:, channel, :, :] = x[:, channel, :, :] - mean[channel]
            x_new[:, channel, :, :] = x[:, channel, :, :] / std[channel]
        return x_new
