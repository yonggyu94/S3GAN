import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

from models.discriminators.resblocks import Block
from models.discriminators.resblocks import OptimizedBlock


class Omniglot_Discriminator(nn.Module):

    def __init__(self, num_features=32, num_classes=0, activation=F.relu):
        super(Omniglot_Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(1, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.l4 = utils.spectral_norm(nn.Linear(num_features * 4, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Linear(num_classes, num_features * 4))

        self.rotate_layer = utils.spectral_norm(nn.Linear(num_features * 4, 4))
        self.pseudo_layer = utils.spectral_norm(nn.Linear(num_features * 4, num_classes))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l4.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l4(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output
    
    
    def discriminator_with_additonal_heads(self, x, y):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.activation(h)
        # Global pooling
        x_rep = torch.sum(h, dim=(2, 3))
        d_logits = self.l4(x_rep)

        is_label_available = (y.sum(dim=1, keepdim=True) > 0.5).float()
        rotation_logits = self.rotate_layer(x_rep)
        pseudo_logits = self.pseudo_layer(x_rep)

        y_pred = torch.argmax(pseudo_logits, dim=1, keepdim=True)
        y_pred_onehot = torch.zeros(x_rep.size(0), self.num_classes)
        y_pred_onehot.scatter_(1, y_pred, 1).cuda()

        y = (1.0 - is_label_available) * y_pred_onehot + is_label_available * y
        # y = torch.argmax(y, dim=1, keepdim=True)
        y = y.detach()

        d_logits += torch.sum(self.l_y(y) * x_rep, dim=1, keepdim=True)
        d_probs = F.sigmoid(d_logits)

        return d_probs, d_logits, rotation_logits, pseudo_logits, is_label_available


class VGG_Discriminator(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(VGG_Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features // 4)
        self.block2 = Block(num_features // 4, num_features // 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features // 2, num_features,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.l4 = utils.spectral_norm(nn.Linear(num_features * 4, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Linear(num_classes, num_features * 4))

        self._initialize()

        self.rotate_layer = utils.spectral_norm(nn.Linear(num_features * 4, 4))
        self.pseudo_layer = utils.spectral_norm(nn.Linear(num_features * 4, num_classes))

    def _initialize(self):
        init.xavier_uniform_(self.l4.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        x_rep = torch.sum(h, dim=(2, 3))
        d_logits = self.l4(x_rep)

        if y is not None:
            d_logits += torch.sum(self.l_y(y) * x_rep, dim=1, keepdim=True)

        return d_logits, x_rep

    def discriminator_with_additonal_heads(self, x, y):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        x_rep = torch.sum(h, dim=(2, 3))
        d_logits = self.l4(x_rep)

        is_label_available = (y.sum(dim=1, keepdim=True) > 0.5).float()
        rotation_logits = self.rotate_layer(x_rep)
        pseudo_logits = self.pseudo_layer(x_rep)

        y_pred = torch.argmax(pseudo_logits, dim=1, keepdim=True)
        y_pred_onehot = torch.zeros(x_rep.size(0), self.num_classes)
        y_pred_onehot.scatter_(1, y_pred, 1).cuda()

        y = (1.0 - is_label_available) * y_pred_onehot + is_label_available * y
        # y = torch.argmax(y, dim=1, keepdim=True)
        y = y.detach()

        d_logits += torch.sum(self.l_y(y) * x_rep, dim=1, keepdim=True)
        d_probs = F.sigmoid(d_logits)

        return d_probs, d_logits, rotation_logits, pseudo_logits, is_label_available

    
class Animal_Discriminator(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(Animal_Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features // 8)
        self.block2 = Block(num_features // 8, num_features // 4,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features // 4, num_features // 2,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features // 2, num_features,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block6 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.l4 = utils.spectral_norm(nn.Linear(num_features * 4, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Linear(num_classes, num_features * 4))

        self._initialize()

        self.rotate_layer = utils.spectral_norm(nn.Linear(num_features * 4, 4))
        self.pseudo_layer = utils.spectral_norm(nn.Linear(num_features * 4, num_classes))

    def _initialize(self):
        init.xavier_uniform_(self.l4.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        # Global pooling
        x_rep = torch.sum(h, dim=(2, 3))
        d_logits = self.l4(x_rep)

        if y is not None:
            d_logits += torch.sum(self.l_y(y) * x_rep, dim=1, keepdim=True)

        return d_logits, x_rep

    def discriminator_with_additonal_heads(self, x, y):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        # Global pooling
        x_rep = torch.sum(h, dim=(2, 3))
        d_logits = self.l4(x_rep)

        is_label_available = (y.sum(dim=1, keepdim=True) > 0.5).float()
        rotation_logits = self.rotate_layer(x_rep)
        pseudo_logits = self.pseudo_layer(x_rep)

        y_pred = torch.argmax(pseudo_logits, dim=1, keepdim=True)
        y_pred_onehot = torch.zeros(x_rep.size(0), self.num_classes)
        y_pred_onehot.scatter_(1, y_pred, 1).cuda()

        y = (1.0 - is_label_available) * y_pred_onehot + is_label_available * y
        # y = torch.argmax(y, dim=1, keepdim=True)
        y = y.detach()

        d_logits += torch.sum(self.l_y(y) * x_rep, dim=1, keepdim=True)
        d_probs = F.sigmoid(d_logits)

        return d_probs, d_logits, rotation_logits, pseudo_logits, is_label_available
