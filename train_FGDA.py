import argparse
import os.path as osp
import os
import sys
from models.JacobianReg import *
import torch.optim as optim
import preprocess.pre_process as prep
from torch.utils.data import DataLoader
import models.lr_schedule as lr_schedule
from preprocess.data_list_2 import ImageList, ImageList_idx
import time
from utils.seed import *
import models.networks as network
from models.SPL import obtain_label
import torch.nn as nn
import torch

def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def add_prefix_to_patch(image_list, prefix):
    for i in range(len(image_list)):
        line = image_list[i]
        line = line.strip('\n')
        abs_path = prefix + line
        image_list[i] = abs_path
    return image_list


def train(config):
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    prefix = config["data_path"]
    dsets["source"] = ImageList_idx(add_prefix_to_patch(open(data_config["source"]["list_path"]).readlines(), prefix), \
                                    transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList_idx(add_prefix_to_patch(open(data_config["target"]["list_path"]).readlines(), prefix), \
                                    transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)
    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(add_prefix_to_patch(open(data_config["test"]["list_path"]).readlines(), prefix), \
                                       transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(add_prefix_to_patch(open(data_config["test"]["list_path"]).readlines(), prefix), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=4)
    class_num = config["network"]["params"]["class_num"]

    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    ad_net = network.GradientDiscriminator(base_network.output_num(), 1024).cuda()

    parameter_f = base_network.get_feature_learner_parameters()
    parameter_c = base_network.get_classifier_parameters()
    parameter_ad = ad_net.get_parameters()

    optimizer_config = config["optimizer"]
    optimizer_d_f = optimizer_config["type"](parameter_ad, **(optimizer_config["optim_params"]))
    optimizer_f = optimizer_config["type"](parameter_f, **(optimizer_config["optim_params"]))
    optimizer_c = optimizer_config["type"](parameter_c, **(optimizer_config["optim_params"]))

    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    mem_label = None

    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                                                 base_network, test_10crop=prep_config["test_10crop"])
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)

            base_network.eval()
            mem_label, log_SPL = obtain_label(dset_loaders['test'], base_network)
            mem_label = torch.from_numpy(mem_label).cuda()
            base_network.train()

            config["out_file"].write(log_SPL + "\n")
            config["out_file"].flush()

        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                                                             "iter_{:05d}_model.pth.tar".format(i)))

        loss_params = config["loss"]
        base_network.train(True)
        ad_net.train(True)

        optimizer_d_f = lr_scheduler(optimizer_d_f, i, **schedule_param)
        optimizer_f = lr_scheduler(optimizer_f, i, **schedule_param)
        optimizer_c = lr_scheduler(optimizer_c, i, **schedule_param)
        optimizer_d_f.zero_grad()
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_source, labels_source, _ = iter_source.next()
        inputs_target, labels_target, tar_idx = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        loss_c_source = nn.CrossEntropyLoss()(outputs_source, labels_source)

        if mem_label is not None:
            pred = mem_label[tar_idx]
            loss_c_target = nn.CrossEntropyLoss()(outputs_target, pred)
        else:
            softmax_target = nn.Softmax(dim=1)(outputs_target).detach()
            pseudo_label = softmax_target.data.max(1, keepdim=False)[1]
            loss_c_target = nn.CrossEntropyLoss()(outputs_target, pseudo_label)

        gradient_source = \
            torch.autograd.grad(outputs=loss_c_source, inputs=features_source, create_graph=True, retain_graph=True,
                                only_inputs=True)[0]
        gradient_target = \
            torch.autograd.grad(outputs=loss_c_target, inputs=features_target, create_graph=True, retain_graph=True,
                                only_inputs=True)[0]

        gradient = torch.cat((gradient_source, gradient_target), dim=0)
        transfer_loss = network.DANN(gradient, ad_net)
        transfer_loss.backward(retain_graph=True)

        optimizer_c.zero_grad()

        reg_fun = JacobianReg()
        JR_reg = reg_fun(features_source, outputs_source)
        JR_loss = (config["jr_coeff"] * (JR_reg))

        loss = loss_c_source + JR_loss

        loss.backward()
        optimizer_c.step()
        optimizer_f.step()
        optimizer_d_f.step()
        optimizer_d_f.zero_grad()
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()


    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

import platform
if __name__ == "__main__":

    data_path = '/Data_SSD/user/dataset_dir'

    parser = argparse.ArgumentParser(description='FGDA')
    parser.add_argument('--method', type=str, default='FGDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50',
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13",
                                 "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default=data_path + '/data/office/dslr_list.txt',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default=data_path + '/data/office/amazon_list.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000,
                        help="interval of two continuous output gd_model")
    parser.add_argument('--output_dir', type=str, default='san',
                        help="output directory of our gd_model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.003, help="learning rate")
    parser.add_argument('--jr_coeff', type=float, default=0.005, help="Jacobian Regularization Coefficient")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    all_seed(1171)
    test = False
    debug = True
    if debug:
        folder = 'debug'
    else:
        folder = 'snapshot'
    config = {}
    config['model_name'] = os.path.basename(sys.argv[0]) + '_' + str(time.time())
    config['data_path'] = data_path
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 100004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = os.path.join(data_path, folder, args.dset, config['model_name']+'_final')
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], config['model_name'] + "_log.txt"), "w")
    config["prep"] = {"test_10crop": False,
                      'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    config["loss"] = {"trade_off": 1.0}
    config["jr_coeff"] = args.jr_coeff
    print(config["output_path"])

    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 36}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": 36}, \
                      "test": {"list_path": args.t_dset_path, "batch_size": 4}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
