# coding='utf-8'
import os
import sys
import argparse
import numpy as np
import time
import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'models'))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'common'))
from batch_reader import BatchReader
import models


def train(prefix, **arg_dict):
    img_size = arg_dict['img_size']
    gpu_num = len(arg_dict["gpu_device"].split(','))
    batch_size = arg_dict["batch_size"]
    common_dict = {"global_step": 1}
    print ("batch_size = %d for gpu_num = %d" % (batch_size, gpu_num))
    if arg_dict["parallel_mode"] == "ModelParallel":
        print ("Working on model parallel.")
        if gpu_num <= 1:
            raise Exception("Model parallel only support more than 2 gpu number")
    elif arg_dict["parallel_mode"] == "DataParallel":
        print ("Working on data parallel")
    else:
        raise Exception("Unsupport parallel mode. see --help")
    # Creat tf_summary writer.
    try:
        from tensorboardX import SummaryWriter
        summary_dir = os.path.join(prefix, "tf_summary")
        if os.path.exists(summary_dir):
            print ("Delete old summary in first.")
            os.system("rm -rf {}".format(summary_dir))
        common_dict["tensorboard_writer"] = SummaryWriter(summary_dir)
        print ("Enable tensorboard summary.")
        print ("Please using 'python -m tensorboard.main --logdir={}'".format(summary_dir))
    except Exception as ex:
        common_dict["tensorboard_writer"] = None
        print ("Disable tensorboard summary. please install tensorboardX in first.")
        print ("Easy to install by 'pip install tensorboardX --user'")
    # batch generator
    _batch_reader = BatchReader(**arg_dict)
    _batch_generator = _batch_reader.batch_generator()
    # net
    model_params = json.loads(arg_dict["model_params"])
    model_params["image_size"] = arg_dict["img_size"]
    model_params["feature_dim"] = arg_dict["feature_dim"]
    model_params["class_num"] = arg_dict["label_num"]
    net =  models.init(arg_dict["model"], gpu_num=gpu_num, model_params=model_params,
                       parallel_mode=arg_dict["parallel_mode"], common_dict=common_dict)
    if arg_dict["parallel_mode"] == "DataParallel":
        net = nn.DataParallel(net)
        net.cuda()
    # print (net)
    if arg_dict["restore_ckpt"]:
        print ("Resotre ckpt from {}".format(arg_dict["restore_ckpt"]))
        net.load_state_dict(torch.load(arg_dict["restore_ckpt"]))
    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=arg_dict['learning_rate'],
                          momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.95)
    # start loop
    print ("Start to training...")
    start_time = time.time()
    display = 100
    loss_list = []
    while not _batch_reader.should_stop():
        #  prepare data
        batch_st = time.time()
        batch = _batch_generator.next()
        datas = batch[0].cuda()
        labels = batch[1].cuda()
        batch_et = time.time()
        #  forward and backward
        loss = net(datas, labels)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lossd = loss.data[0]
        #  display
        loss_list.append(lossd)
        if common_dict["global_step"] % display == 0:
            end_time = time.time()
            cost_time, start_time = end_time - start_time, end_time
            sample_per_sec = int(display * batch_size / cost_time)
            sec_per_step = cost_time / float(display)
            loss_display = np.mean(loss_list)
            lr = optimizer.param_groups[0]['lr']
            print ('[%s] epochs: %d, step: %d, lr: %.5f, loss: %.5f, '\
                   'sample/s: %d, sec/step: %.3f, batch time: %.3fs' % (
                   datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), 
                   _batch_reader.get_epoch(), common_dict["global_step"], lr, loss_display,
                   sample_per_sec, sec_per_step, batch_et - batch_st))
            loss_list = []
            if common_dict["tensorboard_writer"] is not None:
                common_dict["tensorboard_writer"].add_scalar("loss", loss_display,
                                                             common_dict["global_step"])
                common_dict["tensorboard_writer"].add_scalar("sample_per_sec", sample_per_sec,
                                                             common_dict["global_step"])
                common_dict["tensorboard_writer"].add_scalar("lr", lr,
                                                             common_dict["global_step"])
        if common_dict["global_step"] % 10000 == 0:
            # save checkpoint
            checkpoint_path = os.path.join(prefix, 'model.ckpt')
            torch.save(net.state_dict(), checkpoint_path)
            print ("save checkpoint to %s" % checkpoint_path)
        lr_scheduler.step()
        common_dict["global_step"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_paths', type=str, nargs='+', default='')
    parser.add_argument('--working_root', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--max_epoch', type=int, default=100000, help="Training will be stoped in this case.")
    parser.add_argument('--img_size', type=int, default=128, help="The size of input for model")
    parser.add_argument('--feature_dim', type=int, default=512, help="dim of face feature")
    parser.add_argument('--label_num', type=int, default=696877, help="the label num of your training set")
    parser.add_argument('--process_num', type=int, default=20, help="The number of process to preprocess image.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="lr")
    parser.add_argument('--model', type=str, default='SphereFaceNet', help="Model name. Check models.py")
    parser.add_argument('--model_params', type=str, default='{}', help="params for model. dict format")
    parser.add_argument('--restore_ckpt', type=str, help="Resume training from special ckpt.")
    parser.add_argument('--try', type=int, default=0, help="Saving path index")
    parser.add_argument('--gpu_device', type=str, default='7', help="GPU index")
    parser.add_argument('--parallel_mode', type=str, default="DataParallel",
                        help="Support DataParallel and ModelParallel")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    arg_dict = vars(parser.parse_args())
    prefix = '%s/%s/dim%d_size%d_try%d' % (
        arg_dict['working_root'], arg_dict['model'], 
        arg_dict["feature_dim"], arg_dict['img_size'], arg_dict['try'])
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # set up environment
    os.environ['CUDA_VISIBLE_DEVICES']=arg_dict['gpu_device']
    if arg_dict['debug']:
        print ("Enable debug mode!")

    train(prefix, **arg_dict)

if __name__ == "__main__":
    main()

