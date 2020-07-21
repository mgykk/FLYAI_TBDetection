# -*- coding: utf-8 -*-
import argparse
import os
import torch
import cv2
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from PIL import ImageFile
import xml.etree.ElementTree as ET
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from path import MODEL_PATH, DATA_PATH
import pandas as pd
import math
from net import get_model

'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
args = parser.parse_args()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for object in root.findall('object'):
        # object_name = object.find('label').text
        Xmin = int(object.find('bndbox').find('xmin').text)
        Ymin = int(object.find('bndbox').find('ymin').text)
        Xmax = int(object.find('bndbox').find('xmax').text)
        Ymax = int(object.find('bndbox').find('ymax').text)
        boxes.append([Xmin, Ymin, Xmax, Ymax])
    return boxes


def resize_bbox(bbox, in_size, out_size):
    bbox = np.array(bbox.copy())
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def resize_img(img, boxes, min_size=1200, max_size=1600):
    H, W, C = img.shape  # cv
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)

    img = cv2.resize(img, (int(W * scale), int(H * scale)))
    H_, W_, _ = img.shape
    scale = H_ / H
    boxes = resize_bbox(boxes, (H, W), (H_, W_))

    return img, boxes, scale


def rotate_xml(src, box, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    xmin, ymin, xmax, ymax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    rangle = np.deg2rad(angle)
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
    point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
    point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
    point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
    # concat np.array
    concat = np.vstack((point1, point2, point3, point4))
    # change type
    concat = concat.astype(np.int32)
    rx, ry, rw, rh = cv2.boundingRect(concat)
    return rx, ry, rx + rw, ry + rh


def rotate_image(src, boxes, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # map
    rotated = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))))

    boxes = np.array(boxes)
    for i in range(len(boxes)):
        boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3] = rotate_xml(src, boxes[i], angle)
    return rotated, boxes


def img_noise(img_data):
    '''
        添加高斯噪声，均值为0，方差为0.001
    '''
    image = np.array(img_data)
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(0, 0.001 ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out * 255)
    return out


# 自定义获取数据的方式
def data_generate(img, boxes):
    a = np.random.random()
    if a <= 0.2:
        img = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=0.5, sigmaY=0.3)
    if a >= 0.8:
        img = img_noise(img)
    b = np.random.random()
    if b <= 0.2:
        img, boxes, scales = resize_img(img, boxes)
    # c = np.random.random()
    # if c<=0.5:
    #     H_, W_, _ = img.shape
    #     img, params = random_flip(img, x_random=True, return_param=True)
    #     boxes = flip_bbox(boxes, (H_, W_), x_flip=params['x_flip'])
    d = np.random.random()
    if d <= 0.2:
        angle = int(random.uniform(-35, 35))
        img, boxes = rotate_image(img, boxes, angle, scale=1.)
    return img, boxes

class MyDataset(Dataset):
    def __init__(self, root, img_file_list, xml_file_list, transforms=None, data_generate=None):
        self.root = root
        self.transforms = transforms
        self.img_file_list = img_file_list
        self.xml_file_list = xml_file_list
        self.data_generate = data_generate

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_file_list[idx])
        xml_path = os.path.join(self.root, self.xml_file_list[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
        boxes = get_xml(xml_path)

        if self.data_generate is not None:
            img, boxes = self.data_generate(img, boxes)

        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        img = Image.fromarray(img.astype(np.uint8))

        img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.img_file_list)

def collate_fn(batch):
    return tuple(zip(*batch))


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("TBDetection")
        print('download data done...')

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        csv_path = os.path.join(DATA_PATH, 'TBDetection', 'train.csv')
        df = pd.read_csv(csv_path)
        img_file_list = list(df['image_path'].values)
        xml_file_list = list(df['xml_path'].values)
        transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor()])
        train_data = MyDataset(os.path.join(DATA_PATH, 'TBDetection'), img_file_list, xml_file_list,
                               transforms=transform, data_generate = data_generate)
        self.train_loader = DataLoader(dataset=train_data, batch_size=args.BATCH, shuffle=True, collate_fn=collate_fn)
        self.model = get_model()
        #self.model.load_state_dict(torch.load('./model/best.pth'))
        self.model.to(device)
        print('deal with data done...')

    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''
        print('start train...')
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

        lowest_loss = 100
        lowest_batch_loss = 100

        for epoch in range(1, args.EPOCHS+1):
            self.model.train()
            batch_step = 0
            epoch_loss = 0
            for i, (images, targets) in enumerate(self.train_loader):
                batch_step += 1
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                #print('epoch: %d, batch: %d, loss: %f'%(epoch, i, losses))
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                temp_batch_loss = losses.cpu().detach().numpy()
                print('epoch: %d/%d, batch: %d/%d, batch_loss: %f' % (
                    epoch, args.EPOCHS, batch_step, len(self.train_loader), temp_batch_loss))
                epoch_loss += temp_batch_loss

                if temp_batch_loss < lowest_batch_loss:
                    lowest_batch_loss = temp_batch_loss
                    torch.save(self.model.state_dict(), os.path.join(MODEL_PATH, 'best.pth'))

            epoch_loss = epoch_loss / len(self.train_loader)
            print('epoch: %d, loss: %f' % (epoch, epoch_loss))

            lr_scheduler.step()
            if epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
            # 保存模型
                torch.save(self.model.state_dict(), os.path.join(MODEL_PATH, 'best.pth'))
                print("lowest loss: %f" % lowest_loss)


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()