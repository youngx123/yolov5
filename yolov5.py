# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 14:28  2021-12-22
"""
According to the given yolo v5 6.0 version yamL file of any network,
construct the corresponding network structure and load the pre-training weights
img_size = 640
"""
from models.common import *
from torch import nn
import torch
import torch.nn.functional as F
import operator
import yaml

A = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
anchor_grid = np.asarray(A, dtype=np.float32).reshape(3, 1, -1, 1, 1, 2)


class Detect(nn.Module):
    def __init__(self, nc=80, anchors=A, ch=()):  # detection layer
        super().__init__()
        self.nc = nc      # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)          # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.anchor_grid = torch.from_numpy(anchor_grid)
        self.stride = torch.from_numpy(np.array([8., 16., 32.0]))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    # def forward(self, x):
    #     for i in range(self.nl):
    #         x[i] = self.m[i](x[i])
    #     return x

    def forward(self, x):
        for i in range(self.nl):
            device = x[0].device
            x[i] = self.m[i](x[i])
            bs, _, nx, ny = x[i].shape
            x[i] = x[i].sigmoid()
            x[i] = x[i].reshape(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2)

            grid, anchor_grid = self._make_grid(nx, ny, i, device=device)
            x[i][..., 0:2] = (x[i][..., 0:2] * 2 - 0.5 + grid) * self.stride[i]  # xy
            x[i][..., 2:4] = (x[i][..., 2:4] * 2) ** 2 * anchor_grid  # wh
        return x

    def _make_grid(self, nx=20, ny=20, i=0, device=None):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float().to(device)
        anchor_grid = self.anchor_grid[i].clone().view((1, self.na, 1, 1, 2)).expand(
            (1, self.na, ny, nx, 2)).float().to(device)
        return grid, anchor_grid


class YOLOv5_V6(nn.Module):
    def __init__(self, yaml=None):
        super(YOLOv5_V6, self).__init__()
        assert yaml is not None, "input yolov5 framework config file !!!"
        config = self.Read_Yaml(yaml)
        depth_multiple = config['depth_multiple']
        backhead = config['backbone'] + config['head']

        self.b0 = Conv(3, backhead[0][-1], 6, 2, 2)
        self.b1 = Conv(backhead[0][-1], backhead[1][-1], 3, 2)
        self.b2 = C3(backhead[1][-1], backhead[2][-1], max(round(backhead[2][1] * depth_multiple), 1))

        self.b3 = Conv(backhead[2][-1], backhead[3][-1], 3, 2)
        self.b4 = C3(backhead[3][-1], backhead[4][-1], max(round(backhead[4][1] * depth_multiple), 1))

        self.b5 = Conv(backhead[4][-1], backhead[5][-1], 3, 2)
        self.b6 = C3(backhead[5][-1], backhead[6][-1], max(round(backhead[6][1] * depth_multiple), 1))

        self.b7 = Conv(backhead[6][-1], backhead[7][-1], 3, 2)
        self.b8 = C3(backhead[7][-1], backhead[8][-1], max(round(backhead[8][1] * depth_multiple), 1))
        self.b9 = SPPF(backhead[8][-1], backhead[9][-1], 5)

        self.head10 = Conv(backhead[9][-1], backhead[10][-1], 1, 1)
        self.head13 = C3(backhead[10][-1] + backhead[6][-1], backhead[13][-1],
                         max(round(backhead[13][1] * depth_multiple), 1), False)

        self.head14 = Conv(backhead[13][-1], backhead[14][-1], 1, 1)
        self.head17 = C3(backhead[14][-1] + backhead[4][-1], backhead[17][-1],
                         max(round(backhead[17][1] * depth_multiple), 1), False)  # p3

        self.head18 = Conv(backhead[17][-1], backhead[18][-1], 3, 2)
        self.head20 = C3(backhead[18][-1] + backhead[14][-1], backhead[20][-1],
                         max(round(backhead[20][1] * depth_multiple), 1), False)  # p4

        self.head21 = Conv(backhead[20][-1], backhead[21][-1], 3, 2)
        self.head23 = C3(backhead[21][-1] + backhead[10][-1], backhead[23][-1],
                         max(round(backhead[23][1] * depth_multiple), 1), False)  # p5

        self.detect_layer = Detect(ch=(backhead[17][-1], backhead[20][-1], backhead[23][-1]))

    def Read_Yaml(self, yamlfile=None):
        '''
        Load the configuration file and deal with it
        '''
        with open(yamlfile, encoding='ascii', errors='ignore') as f:
            yamldata = yaml.safe_load(f)  # model dict

        self.width_multiple = yamldata['width_multiple']

        for id in range(len(yamldata["backbone"])):
            yamldata["backbone"][id][-1] = self.LayerChannel(yamldata["backbone"][id][-1][0])

        for id in range(len(yamldata["head"]) - 1):
            yamldata["head"][id][-1] = self.LayerChannel(yamldata["head"][id][-1][0])
        return yamldata

    def LayerChannel(self, x, divisor=8):
        if x == 'None' or x == 1:
            return x
        else:
            return math.ceil(x * self.width_multiple / divisor) * divisor

    def forward(self, x):
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        p3 = self.b4(x)  # p3
        x = self.b5(p3)
        p4 = self.b6(x)  # p4
        x = self.b7(p4)
        x = self.b8(x)
        x = self.b9(x)
        # 3 - P3, 5 - P4, 7 - P5
        p5 = self.head10(x)  # p5

        route = F.interpolate(p5, size=(int(p5.shape[2] * 2), int(p5.shape[3] * 2)), mode='nearest')
        x = torch.cat([route, p4], dim=1)
        x = self.head13(x)
        headp4 = self.head14(x)

        route = F.interpolate(headp4, size=(int(headp4.shape[2] * 2), int(headp4.shape[3] * 2)), mode='nearest')
        x = torch.cat([route, p3], dim=1)
        out0 = self.head17(x)

        x = self.head18(out0)
        x = torch.cat([x, headp4], dim=1)
        out1 = self.head20(x)

        x = self.head21(out1)
        x = torch.cat([x, p5], dim=1)
        out2 = self.head23(x)

        out = self.detect_layer([out0, out1, out2])
        return out


def _make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)


def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

    label = '%.2f' % conf
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    return frame


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    ratioh, ratiow = frameHeight / 640, frameWidth / 640
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]  # * detection[4]
            a = detection[4]
            if confidence > 0.5 and detection[4] > 0.5:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        frame = drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
    return frame


if __name__ == '__main__':
    netName = "yolov5s"
    model = YOLOv5_V6(netName + "6.yaml")
    model.eval()
    model_static = model.state_dict()

    pretrained = netName + "6.pt"
    net2 = torch.load(pretrained)

    pretrained_state_dict = net2["model"].state_dict()
    del pretrained_state_dict['model.33.anchors']
    del pretrained_state_dict['model.33.anchor_grid']

    for id, (a, b, namea, nameb) in enumerate(
            zip(pretrained_state_dict.values(), model_static.values(), pretrained_state_dict.keys(),
                model_static.keys())):
        if namea.find('anchor') > -1:
            continue
        if not operator.eq(a.shape, b.shape):
            pass
        else:
            model_static[nameb].copy_(a)

    model.load_state_dict(model_static)

    filepath = "bus.jpg"  # bus
    srcimg0 = cv2.imread(filepath)
    srcimg = cv2.resize(srcimg0, (1280, 1280)) / 255.0
    srcimg = srcimg[np.newaxis, ...]
    srcimg = srcimg.transpose(0, 3, 1, 2)
    srcimg = torch.from_numpy(srcimg).float()
    outs = model(srcimg)

    z = []  # inference output
    for i in range(3):
        y = outs[i].detach().numpy()
        bs, _, _, _, _ = y.shape
        z.append(y.reshape(bs, -1, 85))
    z = np.concatenate(z, axis=1)
    frame = postprocess(srcimg0, z)
    cv2.imwrite(netName + "1.png", frame)
