import sys
import time
import tqdm
import argparse
import torch
from models.SPL import obtain_label
from utils.config import Config
from torch.autograd import Variable
import os
from utils.seed import *

class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i += 1
        return optimizer

# ==============eval
def evaluate(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        probabilities = model_instance.predict(inputs)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_labels) / float(all_labels.size()[0])
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).item() / float(all_labels.size()[0])
    model_instance.set_train(ori_train_state)
    return {'accuracy': accuracy}


def train(config, model_instance, train_source_loader, train_target_loader, test_target_loader,
          group_ratios, max_iter, optimizer, lr_scheduler, eval_interval):
    model_instance.set_train(True)
    print("start train...")

    pseudo_label = None
    mem_label = None
    best_acc = 0.0

    src_loader_len = len(train_source_loader)
    tgt_loader_len = len(train_target_loader)

    min_iter = 900
    model_instance.min_iter = min_iter

    for iter_num in range(max_iter):

        if iter_num % src_loader_len == 0:
            iter_source = iter(train_source_loader)
        if iter_num % tgt_loader_len == 0:
            iter_target = iter(train_target_loader)

        inputs_source, labels_source, _ = iter_source.next()
        inputs_target, _, tar_idx = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num / 5)
        optimizer.zero_grad()

        if model_instance.use_gpu:
            inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(
                inputs_target).cuda(), Variable(labels_source).cuda()
        else:
            inputs_source, inputs_target, labels_source = Variable(inputs_source), Variable(inputs_target), Variable(
                labels_source)

        if mem_label is not None:
            pseudo_label = mem_label[tar_idx]

        train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer, pseudo_label)

        # val
        if iter_num % eval_interval == 0 and iter_num != 0:
            eval_result = evaluate(model_instance, test_target_loader)
            acc = eval_result['accuracy']

            log_str = "iter: {:05d}, precision: {:.5f}".format(iter_num, acc)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)

            # save model
            if acc > best_acc:
                best_acc = acc

                best_model = model_instance.c_net.state_dict()
                torch.save(best_model, os.path.join(config["output_path"], "best_acc_model.pth.tar"))


            if iter_num >= min_iter:
                model_instance.set_train(False)
                mem_label, log_str = obtain_label(test_target_loader, model_instance, log=True)
                mem_label = Variable(torch.from_numpy(mem_label)).cuda()
                model_instance.set_train(True)

                log_str = "iter: {:05d}, ".format(iter_num) + log_str
                config["out_file"].write(log_str + "\n")
                config["out_file"].flush()
                print(log_str)

            if iter_num % 1000 == 0:
                torch.save(model_instance.c_net.state_dict(), os.path.join(config["output_path"], "iter_{:05d}_model.pth.tar".format(iter_num)))


        iter_num += 1

    print('finish train')


def train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer, pseudo_label):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    total_loss = model_instance.get_loss(inputs, labels_source, pseudo_label)
    total_loss.backward()
    optimizer.step()


if __name__ == '__main__':

    all_seed(8497)

    from models.FGDA_MDD import MDD
    from preprocess.data_provider import load_images

    data_path = '/Data_SSD/zhiqiang/Dataset-UDA'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='all sets of configuration parameters',
                        default='./config/dann.yml')
    parser.add_argument('--dataset', default='Office-31', type=str,
                        help='which dataset')
    parser.add_argument('--src_address', default=data_path + '/data/office/amazon_list.txt', type=str,
                        help='address of image list of source dataset')
    parser.add_argument('--tgt_address', default=data_path + '/data/office/dslr_list.txt', type=str,
                        help='address of image list of target dataset')

    parser.add_argument('--jr_coeff', type=float, default=0.005, help="Jacobian Regularization Coefficient")
    parser.add_argument('--r_lambda_1', type=float, default=1.0, help="R for Controlling Lambda 1")
    parser.add_argument('--r_lambda_3', type=float, default=0.5, help="R for Controlling Lambda 3")
    args = parser.parse_args()

    config = {}
    config['method'] ='FGDA_MDD'
    config['model_name'] = os.path.basename(sys.argv[0]) + '_' + str(time.time())
    config['data_path'] = data_path
    config["output_path"] = os.path.join(data_path, 'persistance', args.dataset, config['model_name'])
    if not os.path.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(os.path.join(config["output_path"], config['model_name'] + "_log.txt"), "w")
    print(config["output_path"])

    cfg = Config(args.config)

    source_file = args.src_address
    target_file = args.tgt_address

    if args.dataset == 'Office-31':
        class_num = 31
        width = 1024
        srcweight = 4
        is_cen = False
    elif args.dataset == 'Office-Home':
        class_num = 65
        width = 2048
        srcweight = 2
        is_cen = False

        # Another choice for Office-home:
        # width = 1024
        # srcweight = 3
        # is_cen = True
    else:
        width = -1

    model_instance = MDD(args, base_net='ResNet50', width=width, use_gpu=True, class_num=class_num, srcweight=srcweight)

    train_source_loader = load_images(source_file, batch_size=32, prefix=data_path, is_cen=is_cen)
    train_target_loader = load_images(target_file, batch_size=32,  prefix=data_path, is_cen=is_cen)
    test_target_loader = load_images(target_file, batch_size=32,  prefix=data_path, is_train=False)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)

    train(config, model_instance, train_source_loader, train_target_loader, test_target_loader, group_ratios,
          max_iter=100000, optimizer=optimizer, lr_scheduler=lr_scheduler, eval_interval=100)

