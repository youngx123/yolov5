#pragma once
#include<opencv2/dnn.hpp>
#include<iostream>
#include<string>
using namespace std;
using namespace cv;

class YOLO
{
public:
    YOLO(string onnxpath, float conf=0.5, float nms=0.5, float obj=0.5);
    void Detect(string &imgFile);
    void postProcess(vector<Rect> &boxes, vector<float> &confidences, vector<int> &classIds, Mat &dect_img);
    float Sigmoid(float input);
private:
    string m_onnxPath;
    dnn::Net net;
    const int imgCol = 640;
    const int imgRow = 640;
    const float anchors[3][6] = {
                                {10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, 
                                {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},
                                {116.0, 90.0, 156.0, 198.0, 373.0, 326.0}
                            };
    const float stride[3] = { 8.0, 16.0, 32.0 };
    const string classesFile = "coco.names";
    const int classesNum = 80;
    float confThreshold;
    float nmsThreshold;
    float objThreshold;
};





