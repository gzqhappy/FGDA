import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import math


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in
                              range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in
                       range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class ResNetFc(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                                  {"params": self.bottleneck.parameters(), "lr_mult": 10, 'decay_mult': 2}, \
                                  {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
            else:
                parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                                  {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list

    def get_feature_learner_parameters(self):
        parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.bottleneck.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        return parameter_list

    def get_classifier_parameters(self):
        parameter_list = [{"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        return parameter_list


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, r_lambda_3=0.5):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
        self.r_lambda_3 = r_lambda_3

    def forward(self, x):
        if self.training:
            self.iter_num += (1 * self.r_lambda_3)
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]



class GradientDiscriminator(nn.Module):
    def __init__(self, in_feature, hidden_size, r_lambda_1=1):
        super(GradientDiscriminator, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
        self.batch_norm_1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_size)

        self.r_lambda_1 = r_lambda_1
        self.parameter_list = [{"params": self.parameters(), "lr": 1}]

    def forward(self, x, train_self=False):

        if not train_self:
            if self.training:
                self.iter_num += (1 * self.r_lambda_1)
            self.coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
            x = x * 1.0
            x.register_hook(grl_hook(self.coeff))

        x = self.ad_layer1(x)
        x = self.batch_norm_1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.batch_norm_2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


class GradientDiscriminator_V2(nn.Module):
    def __init__(self, in_feature, hidden_size, num_class, r_lambda_1=0.5):
        super(GradientDiscriminator_V2, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer2_1 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer2_2 = nn.Linear(hidden_size, hidden_size)

        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.ad_layer4 = nn.Linear(hidden_size, num_class)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu2_1 = nn.ReLU()
        self.relu2_2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout2_1 = nn.Dropout(0.5)
        self.dropout2_2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
        self.batch_norm_1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_size)
        self.batch_norm_2_1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm_2_2 = nn.BatchNorm1d(hidden_size)

        self.r_lambda_1 = r_lambda_1
        self.parameter_list = [{"params": self.parameters(), "lr": 1}]

    def forward(self, x, train_self=False):

        if not train_self:
            if self.training:
                self.iter_num += (1 * self.r_lambda_1)
            self.coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
            x = x * 1.0
            x.register_hook(grl_hook(self.coeff))

        x = self.ad_layer1(x)
        x = self.batch_norm_1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.ad_layer2(x)
        x = self.batch_norm_2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        y = self.ad_layer3(x)
        y = self.sigmoid(y)

        y_c = self.ad_layer4(x)
        return y, y_c

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


def DANN(features, ad_net, src_size, tgt_size, outputs=False, train_self=False):
    ad_out, outputs_cla = ad_net(features, train_self)
    dc_target = torch.from_numpy(
        np.array([[1]] * src_size + [[0]] * tgt_size)).float().cuda()  # 1 for source, 0 for target
    d_loss = nn.BCELoss()(ad_out, dc_target)

    if outputs:
        return d_loss, ad_out
    elif train_self:
        return d_loss, outputs_cla
    else:
        return d_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}