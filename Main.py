import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from datetime import date
import pickle

from TrafficSignsDataset import TrafficSignsDataset


# 用于保存log
class Logger(object):
    def __init__(self, dataset):
        self.terminal = sys.stdout
        self.log = open('./YOLO_traffic_log_' +
                        str(date.today()) + '.txt', 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def crt_dir(save_dir, image_save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(image_save_dir):
        os.mkdir(image_save_dir)


def YOLO_traffic(dataset):
    sys.stdout = Logger(dataset)

    if torch.cuda.is_available():
        # torch.device代表将torch.Tensor分配到的设备的对象。
        # torch.device包含一个设备类型（‘cpu’或‘cuda’）和可选的设备序号。
        device = torch.device('cuda')

    # training parameters
    batch_size = 128  # if dataset == 'MNIST' else 64 if dataset == 'CIFAR' else 0
    learning_rate = 0.1
    epochs = 50

    # parameters for Models
    test_every = 5
    print(f"learning rate: {learning_rate}")
    print(f"total epochs of {epochs}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # load model
    # Pytorch Hub 是经过预先训练的模型资料库，旨在促进研究的可重复性。
    YOLO_model = torch.hub.load(  # TODO: https://runebook.dev/zh-CN/docs/pytorch/hub#torch.hub.load
        'ultralytics/yolov5',
        'yolov5s',
        pretrained=False).to(device)

    # Binary Cross Entropy Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Initialise the Optimizers
    optimizer = torch.optim.SGD(YOLO_model.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=0.0001)

    scheduler = scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[0.5*epochs-1, 0.75*epochs-1], gamma=0.1)

    start_time = time.time()

    for epoch in range(epochs):
        YOLO_model.train()

        # 开始计时，记录epoch的开始时间
        epoch_start_time = time.time()

        # FIXME: 没有转换成tensor，还是PIL.Image.Image
        for (data, labels) in tqdm(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = YOLO_model(data)
            trloss = criterion(outputs, labels)
            trloss.backward()
            optimizer.step()

        # test the network every few epochs
        if epoch % test_every == 0 or epoch == epochs - 1:
            accuracy = 0
            YOLO_model.eval()
            with torch.no_grad():
                for (data, labels) in tqdm(YOLO_model):
                    data, labels = data.to(device), labels.to(device)
                    result = YOLO_model(data)
                    teloss = criterion(result, labels)

        scheduler.step(teloss)

        # 结束计时，记录epoch的开始时间
        epoch_endtime = time.time()
        # 每个epoch所花的时间就是：
        per_epoch_ptime = epoch_endtime - epoch_start_time

        print("Epoch %d of %d with %.2f s" %
              (epoch + 1, epochs, per_epoch_ptime))

    torch.save(YOLO_model, 'YOLO_model.pth')

    end_time = time.time()
    total_ptime = end_time - start_time
    print(total_ptime)


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

    # FIXME: 什么是Target Transform ?????
    traffic_dataset = TrafficSignsDataset(
        'GroundTruth.txt', transform=transform)

    YOLO_traffic(traffic_dataset)
