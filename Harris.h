// Harris.h
// 创建者：丁燎原
// 
// Harris角点检测（Harris）
// 步骤：
// 1、利用水平、竖直差分算子求得中心像素点的Ix、Iy，进而求得变换矩阵M。
//    其中，水平差分算子为|-1, 0, 1|，竖直差分算子为| 1, 1, 1|，
//                        |-1, 0, 1|                | 0, 0, 0|
//                        |-1, 0, 1|                |-1,-1,-1|
//    M = |Ix*Ix, Ix*Iy|
//        |Ix*Iy, Iy*Iy|
// 2、对求得的M进行高斯平滑，得到新的M，二维高斯函数为
//    Gauss = exp(-(x*x+y*y)/(2*σ*σ))
// 3、利用高斯平滑后M计算出每一个像素点的角点量R
//    R=det(M)-k*(trace(M)^2)，k一般取0.03-0.06
// 4、对于每一个像素点的角点量R，如果该R大于某一个阈值且该R为选定邻域中最大的，则该像素点为角点
//    阈值越大，则角点数越少；非极大值抑制可以减少角点和提高容忍度
// Harris 具有良好的平移、旋转、光照不变性。但对于尺度、投影、仿射比较敏感。
// 
// 
// 修订历史：
// 2014年12月15日（丁燎原）
//      初始版本
// 2015年4月7日 （丁燎原）
//      添加注释,完善代码（添加错误处理）
// 
#ifndef __HARRIS_H__
#define __HARRIS_H__

#include "Image.h"
#include "Template.h"

// 定义二维矩阵结构体
// | x1 , x2 |
// | x3 , x4 |
typedef struct TwoDMatrix_st
{
    float x1;
    float x2;
    float x3;
    float x4;
} TwoDMatrix;      

class Harris
{
private:
    float threshold;        // R=det(M)-k*(trace(M)^2) R 的阈值    
    int gaussianDimension;  // 高斯滤波模板大小
    float k;                // R=det(M)-k*(trace(M)^2) k 的大小
    int maxDimension;       // 非极大值抑制模板大小
public:
    Harris() {
        threshold = 400000;
        gaussianDimension = 21;
        k = 0.05;
        maxDimension = 21;
    }

    Harris(float threshold, int gaussianDimension, float k, int maxDimension) {
        this->threshold = threshold;
        this->gaussianDimension = gaussianDimension;
        this->k = k;
        this->maxDimension = maxDimension;
    }

    float getThreshold() {
        return threshold;
    }

    void setThreshold(float threshold) {
        this->threshold = threshold;
    }

    int getGaussianDimension() {
        return gaussianDimension;
    }

    void setGaussianDimension(int gaussianDimension) {
        this->gaussianDimension = gaussianDimension;
    }

    float getK() {
        return k;
    }

    void setK(float k) {
        this->k = k;
    }

    int getMaxDimension() {
        return maxDimension;
    }

    void setMaxDimension(int maxDimension) {
        this->maxDimension = maxDimension;
    }

    //int calHessian(Image *inImage, TwoDMatrix *hessianOfImage); // 未调试
    
    // 计算变换矩阵
    int calChangeMatrix(
            Image *inImage,           // 输入图像 
            TwoDMatrix *changeMatrix  // 该图像的变换矩阵，需要用户传入前先开辟空间
    ); 

    int calChangeMatrixOnCPU(
            Image *inImage,           // 输入图像 
            TwoDMatrix *changeMatrix  // 该图像的变换矩阵，需要用户传入前先开辟空间
    );

    // 高斯平滑变换矩阵，并求每一个像素点的R值
    int gaussianChangeMatrix(
            TwoDMatrix *changeMatrix, // 输入图像的变换矩阵
            float *ROfImage,          // 该图像的R值矩阵，需要用户传入前先开辟空间
            int width,                // 输入图像的宽度
            int height                // 输入图像的高度
    ); 

    int gaussianChangeMatrixOnCPU(
            TwoDMatrix *changeMatrix, // 输入图像的变换矩阵
            float *ROfImage,          // 该图像的R值矩阵，需要用户传入前先开辟空间
            int width,                // 输入图像的宽度
            int height                // 输入图像的高度
    );

    // 判断是否为特征点
    int cornerDetection(
            float *ROfImage, // 输入图像的R值矩阵
            int *isCorner,   // 该图像的角点矩阵（如果某点为角点，则在对应位置标1，否则标0）,
                             // 需要用户传入前先开辟空间
            int width,       // 输入图像的宽度
            int height       // 输入图像的高度
    );

    int cornerDetectionOnCPU(
            float *ROfImage, // 输入图像的R值矩阵
            int *isCorner,   // 该图像的角点矩阵（如果某点为角点，则在对应位置标1，否则标0）,
                             // 需要用户传入前先开辟空间
            int width,       // 输入图像的宽度
            int height       // 输入图像的高度
    );

    // 计算Harris角点
    int calHarris(
            Image *inImage,  // 输入图像 
            int *isCorner    // 该图像的角点矩阵（如果某点为角点，则在对应位置标1，否则标0）,
                             // 需要用户传入前先开辟空间 
    );

    int calHarrisOnCPU(
            Image *inImage,  // 输入图像 
            int *isCorner    // 该图像的角点矩阵（如果某点为角点，则在对应位置标1，否则标0）,
                             // 需要用户传入前先开辟空间
    );
};
#endif