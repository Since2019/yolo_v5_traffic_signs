# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
"""
构建Dataset子类，
pytorch读取图片，主要是通过Dataset类，Dataset类作为所有的datasets的基类存在，所有的datasets都需要继承它，类似于c++中的虚基类。

"""

# YOLO:
# X-CENTER Y-CENTER WIDTH HEIGHT
# Width Height: 半径


class TrafficSignsDataset(Dataset):  # 继承Dataset类
    def __init__(self, txt_path, transform=None, target_transform=None):  # 定义txt_path参数
        fh = open(txt_path, 'r')  # 读取txt文件
        imgs = []  # 定义imgs的列表

        # 对每一行进行操作：
        for line in fh:
            line = line.rstrip()  # 默认删除的是空白符（'\n', '\r', '\t', ' '）
            words = line.split(';')  # 默认以空格、换行(\n)、制表符(\t)进行分割，大多是"\"

            # X-CENTER Y-CENTER WIDTH HEIGHT
            # # Width & Height  是半径
            # https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
            imgs.append((words[0], words[1], words[2], words[3]))  # 存放进imgs列表中

        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]  # fn代表图片的路径，label代表标签
        # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1   参考：https://blog.csdn.net/icamera0/article/details/50843172
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)   # 返回图片的长度


def main():
    my_dataset = MyDataset()
