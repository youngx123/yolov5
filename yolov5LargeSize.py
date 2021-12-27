# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 10:19  2021-12-27
"""
According to the given yolo v5 6.0 version yamL file of any network,
construct the corresponding network structure and load the pre-training weights
input image size = 1280
"""
from models.common import *
from torch import nn
import torch
import torch.nn.functional as F
import operator
import yaml

class Detect(nn.Module):
    def __init__(self, anchors=None, ch=None, nc=80):  # detection layer
        super().__init__()
        self.nc = nc      # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.anchor_grid = anchors
        self.nl = len(self.anchor_grid)          # number of detection layers
        self.na = len(self.anchor_grid[0]) // 2  # number of anchors

        self.stride = torch.from_numpy(np.array([8., 16., 32., 64]))
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


class YOLOv5(nn.Module):
    def __init__(self, yaml=None):
        super(YOLOv5, self).__init__()
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
        self.b9 = Conv(backhead[8][-1], backhead[9][-1], 3, 2)

        self.b10 = C3(backhead[9][-1], backhead[10][-1], max(round(backhead[10][1] * depth_multiple), 1))
        self.b11 = SPPF(backhead[10][-1], backhead[11][-1], 5)

        self.head12 = Conv(backhead[11][-1], backhead[12][-1], 1, 1)
        self.head15 = C3(backhead[12][-1] + backhead[8][-1], backhead[15][-1],
                         max(round(backhead[15][1] * depth_multiple), 1), False)

        self.head16 = Conv(backhead[15][-1], backhead[16][-1], 1, 1)
        self.head19 = C3(backhead[16][-1] + backhead[6][-1], backhead[19][-1],
                         max(round(backhead[19][1] * depth_multiple), 1), False)  # p3

        self.head20 = Conv(backhead[19][-1], backhead[20][-1], 3, 1)
        self.head23 = C3(backhead[20][-1] + backhead[4][-1], backhead[23][-1],
                         max(round(backhead[23][1] * depth_multiple), 1), False)  # p4

        self.head24 = Conv(backhead[23][-1], backhead[24][-1], 3, 2)
        self.head26 = C3(backhead[24][-1] + backhead[20][-1], backhead[26][-1],
                         max(round(backhead[26][1] * depth_multiple), 1), False)

        self.head27 = Conv(backhead[26][-1], backhead[27][-1], 3, 2)
        self.head29 = C3(backhead[27][-1] + backhead[16][-1], backhead[29][-1],
                         max(round(backhead[29][1] * depth_multiple), 1), False)

        self.head30 = Conv(backhead[29][-1], backhead[30][-1], 3, 2)
        self.head32 = C3(backhead[30][-1] + backhead[12][-1], backhead[32][-1],
                         max(round(backhead[32][1] * depth_multiple), 1), False)

        self.detect_layer = Detect(anchors=self.anchors,
                                   ch=(backhead[23][-1], backhead[26][-1], backhead[29][-1], backhead[32][-1]))

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

        self.anchors = torch.from_numpy(np.array(yamldata['anchors']))

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
        p5 = self.b8(x)  # p5
        x = self.b9(p5)
        x = self.b10(x)
        x = self.b11(x)
        p6 = self.head12(x)  # p6

        route = F.interpolate(p6, size=(int(p6.shape[2] * 2), int(p6.shape[3] * 2)), mode='nearest')
        x = torch.cat([route, p5], dim=1)  # 14 cat p5
        x = self.head15(x)
        headp5 = self.head16(x)

        route = F.interpolate(headp5, size=(int(headp5.shape[2] * 2), int(headp5.shape[3] * 2)), mode='nearest')
        x = torch.cat([route, p4], dim=1)  # 18 cat p4
        x = self.head19(x)
        headp4 = self.head20(x)

        route = F.interpolate(headp4, size=(int(headp4.shape[2] * 2), int(headp4.shape[3] * 2)), mode='nearest')
        x = torch.cat([route, p3], dim=1)  # 22 cat p3
        out0 = self.head23(x)

        x = self.head24(out0)
        x = torch.cat([x, headp4], dim=1)  # 25 cat headp4
        out1 = self.head26(x)

        x = self.head27(out1)
        x = torch.cat([x, headp5], dim=1)  # 28 cat headp5
        out2 = self.head29(x)

        x = self.head30(out2)
        x = torch.cat([x, p6], dim=1)  # 31 cat p6
        out3 = self.head32(x)

        out = self.detect_layer([out0, out1, out2, out3])
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
    ratioh, ratiow = frameHeight/1280, frameWidth/1280
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
            # a = detection[4]
            if confidence > 0.2 and detection[4] > 0.2:
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.5)
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
    netName = "yolov5s6"
    model = YOLOv5(netName + ".yaml")
    model.eval()
    # a = torch.rand(1, 3, 1280, 1280)
    # o = model(a)
    # model_static = model.state_dict()

    # pretrained = netName + ".pt"
    # net2 = torch.load(pretrained)
    #
    # pretrained_state_dict = net2["model"].state_dict()
    # del pretrained_state_dict['model.33.anchors']
    # del pretrained_state_dict['model.33.anchor_grid']

    # for id, (a, b, namea, nameb) in enumerate(
    #         zip(pretrained_state_dict.values(), model_static.values(), pretrained_state_dict.keys(),
    #             model_static.keys())):
    #     # print(id, a.shape, b.shape, namea, nameb)
    #     if namea.find('anchor') > -1:
    #         continue
    #     if not operator.eq(a.shape, b.shape):
    #         pass
    #     else:
    #         model_static[nameb].copy_(a)
    #
    # model.load_state_dict(model_static)
    # torch.save(model.state_dict(), netName+ ".pth")

    model.load_state_dict(torch.load(netName+ ".pth"))

    filepath = "zidane.jpg"  # bus
    srcimg0 = cv2.imread(filepath)
    srcimg = cv2.resize(srcimg0, (1280, 1280)) / 255.0
    srcimg = srcimg[np.newaxis, ...]
    srcimg = srcimg.transpose(0, 3, 1, 2)
    srcimg = torch.from_numpy(srcimg).float()
    outs = model(srcimg)
    #
    # stride = np.array([8., 16., 32.])
    # # outs = list(outs)  # #[batchsize, 3*(4coord +1object_score + classNum), grid_size1~3, grid_size1~3]
    z = []  # inference output
    for i in range(len(outs)):
        y = outs[i].detach().numpy()
        bs, _, _, _, _ = y.shape
        z.append(y.reshape(bs, -1, 85))
    z = np.concatenate(z, axis=1)
    frame = postprocess(srcimg0, z)
    cv2.imwrite(netName + "61.png", frame)
