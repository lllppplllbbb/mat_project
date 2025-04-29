import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.vggNet import VGGFeatureExtractor


class PerceptualLoss(nn.Module):
    def __init__(self,
                 layer_weights={'relu4_2': 1/4, 'relu5_4': 1/2},
                 vgg_type='vgg19',
                 use_input_norm=True,
                 use_pcp_loss=True,
                 use_style_loss=False,
                 norm_img=True,
                 criterion='l1'):
        super().__init__()
        self.norm_img = norm_img
        self.use_pcp_loss = use_pcp_loss
        self.use_style_loss = use_style_loss
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion not supported.')

    def forward(self, x, gt):
        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5

        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.use_pcp_loss:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
        else:
            percep_loss = torch.tensor(0.0, device=x.device)

        # calculate style loss
        if self.use_style_loss:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) * self.layer_weights[k]
        else:
            style_loss = torch.tensor(0.0, device=x.device)

        return percep_loss, x_features

    def _gram_mat(self, x):
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

