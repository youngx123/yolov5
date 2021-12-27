#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<vector>
#include "yolo.h"

YOLO::YOLO(string onnxpath, float conf, float nms, float obj)
    :m_onnxPath(onnxpath), confThreshold(conf), nmsThreshold(nms), objThreshold(obj)
{
    this->net = dnn::readNetFromONNX(this->m_onnxPath);
    if(!this->net.empty())
    {
        printf("load %s success !!!\n", this->m_onnxPath.c_str());
    }

}


void YOLO::postProcess(vector<Rect> &boxes, vector<float> &confidences, vector<int> &classIds, Mat &dect_img)
/*
检测结果后处理，并显示
*/
{
    //NMS 抑制
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

    //draw bbox on raw image
     for(int i: indices)
     {
        Rect b = boxes.at(i);
        float c = confidences.at(i);
        int catid = classIds.at(i);
        rectangle(dect_img,Point(b.x, b.y), Point(b.x+ b.width , b.y+b.height), Scalar(0,0,255), 2);

        string label_score = format("%.2f", c);
        putText(dect_img,label_score,Point(b.x, b.y), cv::FONT_HERSHEY_SIMPLEX, 1,Scalar(0,0,255), 2);
        cout<<i<<endl;
     }
    imwrite("result.png", dect_img);
}


float YOLO::Sigmoid(float input)
{
    input = 1.0 / (1 + expf(-input));
    return input;
}

void YOLO::Detect(string &imgFile)
{
    Mat dect_img = imread(imgFile);
    if(!dect_img.empty())
    {
        printf("load image %s success !!!\n", imgFile.c_str());
    }

    Mat inputImg ;
    inputImg = dnn::blobFromImage(dect_img, 1/255.0, Size(640,640), Scalar(0,0,0), true, false);
    
    int originCols = dect_img.cols;
    int originRows = dect_img.rows;

    vector<Mat> outPuts;
    this->net.setInput(inputImg);
    this->net.forward(outPuts, this->net.getUnconnectedOutLayersNames());

	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
    
    const float ratioh = (float)originRows / this->imgRow;
    const float ratiow = (float)originCols / this->imgCol;

    for(int i=1; i<3; i++) //三个尺度
    {
        int num_grid_x = (int)(this->imgCol / this->stride[i]);
        int num_grid_y = (int)(this->imgRow / this->stride[i]);

        int row_index = 0;
        for (int q = 0; q < 3; q++)    ///anchor 3个anchor
		{
			const float anchor_w = this->anchors[i][q * 2];
			const float anchor_h = this->anchors[i][q * 2 + 1];

            for(int k=0; k<num_grid_y; k++)
            {
                for(int j=0; j<num_grid_x; j++)
                {
                    // float* pdata = (float*)outs[0].data + row_index * 85;  // outPuts[i].row(row_index).colRange(0, outPuts[i].cols);
                    // float box_score = sigmoid_x(pdata[4]); 
                    Mat result = outPuts[i].row(row_index).colRange(0, outPuts[i].cols); // 85 number result

                    float* pdata = (float*)result.data ;
                    float box_score = this->Sigmoid(pdata[4]);
                    if (box_score > this->objThreshold)
                        {
                            double max_class_socre;
                            Point classIdPoint;
                            Mat catScore = outPuts[i].row(row_index).colRange(5, outPuts[i].cols);
                            minMaxLoc(catScore, 0, &max_class_socre, 0, &classIdPoint);
                            max_class_socre = this->Sigmoid( (float)max_class_socre );
                            if(max_class_socre> this->confThreshold)
                            {
                                float cx = (this->Sigmoid(pdata[0]) * 2.f - 0.5f + j) * this->stride[i];  // cx
                                float cy = (this->Sigmoid(pdata[1]) * 2.f - 0.5f + k) * this->stride[i];   // cy
                                float w = powf(this->Sigmoid(pdata[2]) * 2.f, 2.f) * anchor_w;   // w
                                float h = powf(this->Sigmoid(pdata[3]) * 2.f, 2.f) * anchor_h ;  // h

                                int left = (cx - 0.5*w) * ratiow;
                                int top = (cy - 0.5*h) * ratioh;   ///原图坐标

                                classIds.push_back(classIdPoint.x);
                                confidences.push_back(max_class_socre);
                                boxes.push_back( Rect(left, top, (int)(w*ratiow), (int)(h*ratioh) ) );
                            }
                        }
                    row_index++;
                }
            }
        }
    }
    
    // post process
    this->postProcess(boxes, confidences, classIds, dect_img);
}

int main()
{
    string onnx_path = "./cpp_yolo/yolo_onnx/yolov5s.onnx";
    string dect_img = "./cpp_yolo/bus.jpg";

    YOLO yoloNet(onnx_path);
    yoloNet.Detect(dect_img);
}