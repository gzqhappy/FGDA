from models import backbone as backbone
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from models.JacobianReg import *


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


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


class GradientReverseLayer(torch.autograd.Function):
    def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0, r_lambda_3=0.5):
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter
        self.r_lambda_3 = r_lambda_3

    def forward(self, input):
        self.iter_num += (1*self.r_lambda_3)
        output = input * 1.0
        return output

    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                    self.high_value - self.low_value) + self.low_value)
        return -self.coeff * grad_output


class MDDNet(nn.Module):

    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31, r_lambda_3=0.5):
        super(MDDNet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = GradientReverseLayer(r_lambda_3=r_lambda_3)

        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim),
                                      nn.BatchNorm1d(bottleneck_dim), nn.ReLU(),
                                      nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)

        ## collect parameters
        self.parameter_list = [{"params": self.base_network.parameters(), "lr": 0.1},
                               {"params": self.bottleneck_layer.parameters(), "lr": 1},
                               {"params": self.classifier_layer.parameters(), "lr": 1},
                               {"params": self.classifier_layer_2.parameters(), "lr": 1}]

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)

        # adversarial feature
        features_adv = self.grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)

        # normal feature
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv


class GradientClassifier(nn.Module):
    def __init__(self, in_feature, hidden_size, class_num):
        super(GradientClassifier, self).__init__()
        self.class_num = class_num
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, class_num)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_size)

        self.parameter_list = [{"params": self.parameters(), "lr": 1}]

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.batch_norm_1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.batch_norm_2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        return y

    def output_num(self):
        return self.class_num

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


def DANN(features, ad_net, src_size, tgt_size):
    ad_out = ad_net(features)
    dc_target = torch.from_numpy(np.array([[1]] * src_size + [[0]] * tgt_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


class GradientDiscriminator(nn.Module):
    def __init__(self, in_feature, hidden_size, r_lambda_1=0.5):
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


    def forward(self, x):
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


class MDD(object):

    def __init__(self, args, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num, args.r_lambda_3)
        self.discriminator = GradientDiscriminator(width, 1024, args.r_lambda_1)
        self.gradient_classifier = GradientClassifier(width, width, class_num)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
            self.discriminator = self.discriminator.cuda()
            self.gradient_classifier = self.gradient_classifier.cuda()
        self.srcweight = srcweight
        self.min_iter = None
        self.jr_coeff = args.jr_coeff

    def get_loss(self, inputs, labels_source, pseudo_label):
        inputs.requires_grad = True
        class_criterion = nn.CrossEntropyLoss()

        inputs_src = inputs.narrow(0, 0, labels_source.size(0))
        features_src, outputs_src, softmax_outputs_src, outputs_adv_src = self.c_net(inputs_src)
        loss_c_src = class_criterion(outputs_src, labels_source)

        inputs_tgt = inputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        features_tgt, outputs_tgt, softmax_outputs_tgt, outputs_adv_tgt = self.c_net(inputs_tgt)
        if pseudo_label is not None:
            loss_c_tgt = class_criterion(outputs_tgt, pseudo_label)
        else:
            pred = softmax_outputs_tgt.data.detach().max(1, keepdim=False)[1]
            pred = Variable(pred)
            loss_c_tgt = class_criterion(outputs_tgt, pred)

        gradient_source = \
            torch.autograd.grad(outputs=loss_c_src, inputs=features_src, create_graph=True, retain_graph=True,
                                only_inputs=True)[0]
        gradient_target = \
            torch.autograd.grad(outputs=loss_c_tgt, inputs=features_tgt, create_graph=True, retain_graph=True,
                                only_inputs=True)[0]

        gradients = torch.cat((gradient_source, gradient_target), dim=0)
        df_loss = DANN(gradients, self.discriminator, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        df_loss.backward(retain_graph=True)
        self.c_net.classifier_layer.zero_grad()

        total_loss = loss_c_src


        if self.iter_num >= self.min_iter:
            # MDD
            target_adv_src = outputs_src.max(1)[1]
            target_adv_tgt = outputs_tgt.max(1)[1]

            classifier_loss_adv_src = class_criterion(outputs_adv_src,
                                                      target_adv_src)
            logloss_tgt = torch.log(1 - F.softmax(outputs_adv_tgt, dim=1))
            classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

            transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt
            total_loss += transfer_loss

            reg_fun = JacobianReg()
            JR_reg = reg_fun(features_src, outputs_src)
            total_loss += (self.jr_coeff * (JR_reg))

        self.iter_num += 1

        return total_loss

    def predict(self, inputs, typical_output=False):
        feature, output, softmax_outputs, _ = self.c_net(inputs)

        if typical_output:
            return feature, output
        else:
            return softmax_outputs

    def get_parameter_list(self):
        p_0 = self.c_net.parameter_list
        p_1 = self.discriminator.parameter_list
        p_2 = self.gradient_classifier.parameter_list
        p_0.extend(p_1)
        p_0.extend(p_2)
        return p_0

    def set_train(self, mode):
        self.c_net.train(mode)
        self.discriminator.train(mode)
        self.is_train = mode
