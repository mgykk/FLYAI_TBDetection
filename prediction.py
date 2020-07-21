# -*- coding: utf-8 -*
from flyai.framework import FlyAI
from net import get_model
from torchvision import transforms
import torch
import os
from path import MODEL_PATH
from PIL import Image

# 判断gpu是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
transform = transforms.Compose([transforms.ToTensor()])



def get_return(boxes, labels, scores, image_name, label_id_name):
    ''' 输入示例：
    :param boxes: [[735.097,923.59283,770.1911,998.49335], [525.39496,535.89667,578.4822,589.8431 ]]]   box格式[Xmin, Ymin, Xmax, Ymax]
    :param labels: [1,1]
    :param scores: [0.2， 0.3]
    :param image_name: 0.jpg
    :param label_id_name: {1: 'TBbacillus'}  label id 到 label name的映射

    :return:  [{"image_name": '0.jpg', "label_name": 'TBbacillus', "bbox": [735, 923, 770-735, 998-923], "confidence": 0.2},
                {"image_name": '0.jpg', "label_name": 'TBbacillus', "bbox": [525, 535, 578-525, 589-535], "confidence": 0.3}]
                返回 box的格式为[xmin, ymin, width, height]
    最终评估方式采用coco数据集的map，具体可参考 https://cocodataset.org/
    '''
    result = []
    if len(boxes) == len(labels) == len(scores):
        for i in range(len(boxes)):
            box = boxes[i] # [Xmin, Ymin, Xmax, Ymax]
            bbox = [int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])] # [xmin, ymin, width, height]
            label_name = label_id_name[labels[i]]
            confidence = scores[i]
            ann = {"image_name": image_name, "label_name": label_name, "bbox": bbox, "confidence": confidence}
            result.append(ann)
    return result



class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        self.model = get_model()
        print('load from best.pth...')
        self.model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'best.pth')))
        self.model.to(device)
        print('load model done...')

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"image_path": "./data/input/image/0.jpg"}
        :return: 具体见 get_return 方法
        '''
        image_name = image_path.split('/')[-1]

        img = Image.open(image_path)
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        self.model.eval()
        preds = self.model(img)
        preds = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds]
        preds = [{k: v.data.cpu().numpy() for k, v in t.items()} for t in preds][0]

        result = get_return(preds['boxes'], preds['labels'], preds['scores'], image_name, label_id_name={1: 'TBbacillus'})
        return result
