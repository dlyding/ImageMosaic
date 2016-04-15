#include "Harris.h"
#include "ErrorCode.h"
#include <iostream>
#include <cmath>
#include <stdio.h>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

/*// 未调试
// **************************************************************************
__global__ void _calHessian(ImageCuda inimg, TwoDMatrix *d_hessianOfImage);

__global__ void _calHessian(ImageCuda inimg, TwoDMatrix *d_hessianOfImage)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    int index = r * inimg.imgMeta.width + c;

    // 舍弃边界,使边界元素的 Hassian 矩阵值为0
    if (c == 0 || c == inimg.imgMeta.width - 1 || r == 0 || r == inimg.imgMeta.height - 1) {
        d_hessianOfImage[index].x1 = 0;
        d_hessianOfImage[index].x2 = 0;
        d_hessianOfImage[index].x3 = 0;
        d_hessianOfImage[index].x4 = 0;
        return;
    }

    // | 8, 4, 5 |
    // | 3, 0, 1 |
    // | 7, 2, 6 |
    int x0 = r * inimg.pitchBytes + c;
    //int x1 = x0 + 1;
    int x2 = x0 + inimg.pitchBytes;
    //int x3 = x0 - 1;
    int x4 = x0 - inimg.pitchBytes;
    //int x5 = x4 + 1;
    //int x6 = x2 + 1;
    //int x7 = x2 - 1;
    //int x8 = x4 - 1;

    d_hessianOfImage[index].x1 = inimg.imgMeta.imgData[x0 + 1] + inimg.imgMeta.imgData[x0 - 1] 
                                 - 2 * inimg.imgMeta.imgData[x0];
    d_hessianOfImage[index].x4 = inimg.imgMeta.imgData[x2] + inimg.imgMeta.imgData[x4]
                                 - 2 * inimg.imgMeta.imgData[x0];
    d_hessianOfImage[index].x2 = ((inimg.imgMeta.imgData[x4 - 1] + inimg.imgMeta.imgData[x2 + 1])
                                 - (inimg.imgMeta.imgData[x4 + 1] + inimg.imgMeta.imgData[x2 - 1]))/4.0;
    d_hessianOfImage[index].x3 = d_hessianOfImage[index].x2;
}

int Harris::calHessian(Image *inImage, TwoDMatrix *hessianOfImage)
{
    if (inImage == NULL || hessianOfImage == NULL)
        return NULL_POINTER;

    TwoDMatrix *d_hessianOfImage;
    int size = inImage->height * inImage->width * sizeof(TwoDMatrix);

    ImageBasicOp::copyToCurrentDevice(inImage);

    cudaMalloc((void **)&d_hessianOfImage, size);

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    ImageBasicOp::roiSubImage(inImage, &insubimgCud);

    dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    _calHessian<<<gridsize,blocksize>>>(insubimgCud, d_hessianOfImage);

    cudaMemcpy(hessianOfImage, d_hessianOfImage, size, cudaMemcpyDeviceToHost);

    cudaFree(d_hessianOfImage);

    return 0;
}*/

// ***************************************************************************

// ***************************************************************************
__global__ void _calChangeMatrix(
        ImageCuda inimg,              // 输入图像
        TwoDMatrix *d_changeMatrix,   // 变换矩阵
        TemplateCuda tplx,            // 水平差分算子模板
        TemplateCuda tply             // 竖直差分算子模板
);

// 初始化奇数维正方形模板
int initialTemplate(
        Template *inTl,    // 需初始化的模板，传入前需要先new
        int dimension,     // 模板维度
        float *tlData      // 模板附带的数据，传入前应有数据
);

__global__ void _calChangeMatrix(ImageCuda inimg, TwoDMatrix *d_changeMatrix, 
                                 TemplateCuda tplx, TemplateCuda tply)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
        return;

    int index = r * inimg.imgMeta.width + c;

    // 舍弃边界,使边界元素的变换矩阵值为0（由于模板大小为3，所以舍弃第一行、第一列、最后一行、最后一列）
    if (c == 0 || c == inimg.imgMeta.width - 1 || r == 0 || r == inimg.imgMeta.height - 1) {
        d_changeMatrix[index].x1 = 0;
        d_changeMatrix[index].x2 = 0;
        d_changeMatrix[index].x3 = 0;
        d_changeMatrix[index].x4 = 0;
        return;
    }

    // 初始化Ix，Iy
    float Ix = 0;
    float Iy = 0;
    int cx, cy;

    // 通过模板计算Ix
    for(int i = 0; i < tplx.tplMeta.count; i++) {
        cx = c + tplx.tplMeta.tplData[2 * i];
        cy = r + tplx.tplMeta.tplData[2 * i + 1];
        Ix += ((float)inimg.imgMeta.imgData[cy * inimg.pitchBytes + cx] * tplx.attachedData[i]);
    }

    // 通过模板计算Iy
    for(int i = 0; i < tply.tplMeta.count; i++) {
        cx = c + tply.tplMeta.tplData[2 * i];
        cy = r + tply.tplMeta.tplData[2 * i + 1];
        //Iy += ((float)inimg.imgMeta.imgData[cy * inimg.pitchBytes + cx] * ATTACHED_DATA(&tply)[i]);
        //ATTACHED_DATA(&tply)该函数有问题
        
        Iy += ((float)inimg.imgMeta.imgData[cy * inimg.pitchBytes + cx] * tply.attachedData[i]);
    }

    // 计算变换矩阵
    d_changeMatrix[index].x1 = Ix * Ix;
    d_changeMatrix[index].x2 = Ix * Iy;
    d_changeMatrix[index].x3 = Ix * Iy;
    d_changeMatrix[index].x4 = Iy * Iy;
}

int initialTemplate(Template *inTl, int dimension, float *tlData)
{
	//下次要注意每修改一点，就要测试一下，免得全改完，出错了，不知道bug出在哪儿
   /* if (inTl == NULL || tlData == NULL)
        return NULL_POINTER;*/

    if (inTl == NULL)
        return NULL_POINTER;

    int errcode;
    // 为模板申请空间，模板中点数为模板维度的平方
    errcode = TemplateBasicOp::makeAtHost(inTl, dimension * dimension);
    if (errcode != NO_ERROR)
        return errcode;
    // 根据维度初始化模板中的点的坐标
    for (int i = 0; i < dimension * dimension; i++) {
        inTl->tplData[2 * i] = i % dimension - (dimension - 1) / 2;
        inTl->tplData[2 * i + 1] = i / dimension - (dimension - 1) / 2;
    }

    // 为模板添加附加数据（如果tlData为空，则不附加）
    if (tlData != NULL) {
        TemplateCuda *inTlCuda = TEMPLATE_CUDA(inTl);
        for (int i = 0; i < dimension * dimension; i++) {
            inTlCuda->attachedData[i] = tlData[i];
        }
    }
    
    return NO_ERROR;
}

int Harris::calChangeMatrix(Image *inImage, TwoDMatrix *changeMatrix)
{
    if (inImage == NULL || changeMatrix == NULL)
        return NULL_POINTER;

    int errcode;

    Template *changeX;
    Template *changeY;
    // 水平差分算子
    float dx[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    // 竖直差分算子
    float dy[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    // 创建模板
    errcode = TemplateBasicOp::newTemplate(&changeX);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = TemplateBasicOp::newTemplate(&changeY);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        return errcode;
    }
    // 初始化模板
    errcode = initialTemplate(changeX, 3, dx);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        TemplateBasicOp::deleteTemplate(changeY);
        return errcode;
    }
    errcode = initialTemplate(changeY, 3, dy);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        TemplateBasicOp::deleteTemplate(changeY);
        return errcode;
    }

    // 获取模板的cuda类型数据
    TemplateCuda *changeXCuda = TEMPLATE_CUDA(changeX);
    TemplateCuda *changeYCuda = TEMPLATE_CUDA(changeY);
/*    for (int i = 0; i < 9; i++) {
        cout<<changeX->tplData[2 * i]<<" ";
        cout<<changeX->tplData[2 * i + 1]<<" ";
        cout<<changeXCuda->attachedData[i];
        cout<<endl;
    }*/
    // 将模板拷贝到设备中
    errcode = TemplateBasicOp::copyToCurrentDevice(changeX);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        TemplateBasicOp::deleteTemplate(changeY);
        return errcode;
    }
    errcode = TemplateBasicOp::copyToCurrentDevice(changeY);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        TemplateBasicOp::deleteTemplate(changeY);
        return errcode;
    }
    
    // 将图像拷贝到设备中
    errcode = ImageBasicOp::copyToCurrentDevice(inImage);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        TemplateBasicOp::deleteTemplate(changeY);
        return errcode;
    }

    // 为变换矩阵开辟空间
    TwoDMatrix *d_changeMatrix;
    int size = inImage->height * inImage->width * sizeof(TwoDMatrix);
    cudaMalloc((void **)&d_changeMatrix, size);

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inImage, &insubimgCud);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        TemplateBasicOp::deleteTemplate(changeY);
        cudaFree(d_changeMatrix);
        return errcode;
    }

    dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    _calChangeMatrix<<<gridsize,blocksize>>>(insubimgCud, d_changeMatrix, *changeXCuda, *changeYCuda);

    cudaMemcpy(changeMatrix, d_changeMatrix, size, cudaMemcpyDeviceToHost);
    errcode = ImageBasicOp::copyToHost(inImage);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        TemplateBasicOp::deleteTemplate(changeY);
        cudaFree(d_changeMatrix);
        return errcode;
    }

    TemplateBasicOp::deleteTemplate(changeX);
    TemplateBasicOp::deleteTemplate(changeY);

    cudaFree(d_changeMatrix);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return errcode;
}

// ***************************************************************************

// ***************************************************************************

// 初始化高斯模板的附加数据
int initialGaussian(
        int dimension,   // 高斯模板维度
        float *value     // 高斯模板附加数据数组，传入前需要开辟空间
);

__global__ void _gaussianChangeMatrix(
        TwoDMatrix *d_changeMatrix,     // 变换矩阵
        float *d_ROfImage,              // R值矩阵
        TemplateCuda tl,                // 高斯模板
        int width,                      // 图像宽度
        int height,                     // 图像高度
        float k,                        // k值
        int dimension                   // 高斯模板维度
);

// dimension = 6 * sigma + 1
int initialGaussian(int dimension, float *value)
{
    if (value == NULL)
        return NULL_POINTER;
    // 通过维度算出sigma
    float sigma = (dimension - 1) / 6.0; //注意整形和浮点型之间的转换
    float sum = 0;
    // 计算高斯模板附加数据，并计算总和，方便归一化
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            value[i * dimension + j] = exp(-((i - dimension / 2) * (i - dimension / 2) + 
                                       (j - dimension / 2) * (j - dimension / 2)) / (2 * sigma * sigma));
            sum += value[i * dimension + j];  
        }
    }
    // 归一化
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            value[i * dimension + j] /= sum;
        }
    }

    return NO_ERROR;
}

__global__ void _gaussianChangeMatrix(TwoDMatrix *d_changeMatrix, float *d_ROfImage, 
                                      TemplateCuda tl, int width, int height, float k, int dimension)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= width || r >= height)
        return;

    int index = r * width + c;

    // 舍弃边缘
    int ignore = (dimension + 1) / 2; 
    if (c < ignore || c >= width - ignore || r < ignore || r >= height - ignore) {
        d_ROfImage[index] = 0;
        return;
    }

    // 变换矩阵局部变量
    TwoDMatrix temp;
    temp.x1 = 0;
    temp.x2 = 0;
    temp.x3 = 0;
    temp.x4 = 0;
    int cx, cy;

    // 高斯平滑
    for(int i = 0; i < tl.tplMeta.count; i++) {
        cx = c + tl.tplMeta.tplData[2 * i];
        cy = r + tl.tplMeta.tplData[2 * i + 1];
        temp.x1 += (d_changeMatrix[cy * width + cx].x1 * tl.attachedData[i]);
        temp.x2 += (d_changeMatrix[cy * width + cx].x2 * tl.attachedData[i]);
        temp.x3 += (d_changeMatrix[cy * width + cx].x3 * tl.attachedData[i]);
        temp.x4 += (d_changeMatrix[cy * width + cx].x4 * tl.attachedData[i]);
    }

    // 计算该像素点的R值
    d_ROfImage[index] = temp.x1 * temp.x4 - temp.x2 * temp.x3 - k * (temp.x1 + temp.x4) * (temp.x1 + temp.x4);
}

int Harris::gaussianChangeMatrix(TwoDMatrix *changeMatrix, float *ROfImage, int width, int height)
{
    if (changeMatrix == NULL || ROfImage == NULL)
        return NULL_POINTER;

    int errcode;
    // 高斯模板附加数据数组
    float *gaussianValue = new float[gaussianDimension * gaussianDimension];
    errcode = initialGaussian(gaussianDimension, gaussianValue);
    if (errcode != NO_ERROR) {
        delete []gaussianValue;
        return errcode;
    }

    // 初始化高斯模板
    Template *gaussianTl;
    errcode = TemplateBasicOp::newTemplate(&gaussianTl);
    if (errcode != NO_ERROR) {
        delete []gaussianValue;
        return errcode;
    }

    errcode = initialTemplate(gaussianTl, gaussianDimension, gaussianValue);
    if (errcode != NO_ERROR) {
        delete []gaussianValue;
        TemplateBasicOp::deleteTemplate(gaussianTl);
        return errcode;
    }
    // 将高斯模板拷贝到设备
    TemplateCuda *gaussianTlCuda = TEMPLATE_CUDA(gaussianTl);
    errcode = TemplateBasicOp::copyToCurrentDevice(gaussianTl);
    if (errcode != NO_ERROR) {
        delete []gaussianValue;
        TemplateBasicOp::deleteTemplate(gaussianTl);
        return errcode;
    }

    // 为变换矩阵开辟设备空间，并将其拷贝到设备
    int size1 = width * height * sizeof(TwoDMatrix);
    TwoDMatrix *d_changeMatrix;
    cudaMalloc((void **)&d_changeMatrix, size1);
    cudaMemcpy(d_changeMatrix, changeMatrix, size1, cudaMemcpyHostToDevice);

    // 为R值矩阵开辟设备空间
    float *d_ROfImage;
    int size2 = width * height * sizeof(float);
    cudaMalloc((void **)&d_ROfImage, size2);

    dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height + blocksize.y - 1) / blocksize.y;

    _gaussianChangeMatrix<<<gridsize,blocksize>>>(d_changeMatrix, d_ROfImage, *gaussianTlCuda, 
                                                  width, height, this->k, this->gaussianDimension);

    cudaMemcpy(ROfImage, d_ROfImage, size2, cudaMemcpyDeviceToHost);

    delete []gaussianValue;
    TemplateBasicOp::deleteTemplate(gaussianTl);
    cudaFree(d_changeMatrix);
    cudaFree(d_ROfImage);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return errcode;
}

// ***************************************************************************

// ***************************************************************************

__global__ void _cornerDetection(
        float *d_ROfImage,     // R值矩阵
        int *d_isCorner,       // 判断矩阵
        TemplateCuda tl,       // 非极大值抑制模板
        int width,             // 图像宽度
        int height,            // 图像高度
        float threshold        // 阈值
);

__global__ void _cornerDetection(float *d_ROfImage, int *d_isCorner, 
                                 TemplateCuda tl, int width, int height, float threshold)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= width || r >= height)
        return;

    int index = r * width + c;

    d_isCorner[index] = 0;
    float max = 0;
    int cx, cy;
    // 判断是否为角点
    if (d_ROfImage[index] > threshold) {
        for(int i = 0; i < tl.tplMeta.count; i++) {
            cx = c + tl.tplMeta.tplData[2 * i];
            cy = r + tl.tplMeta.tplData[2 * i + 1];
            if ((cx < width) && (cx >= 0) && (cy < height) && (cy >= 0)) {
                max = (d_ROfImage[cy * width + cx] > max) ? d_ROfImage[cy * width + cx] : max;
            }
        }
        if (max == d_ROfImage[index]) {
            d_isCorner[index] = 1;
        }
        // 非极大值抑制
        else {
            d_isCorner[index] = 0;
        }
    }
    else {
        d_isCorner[index] = 0;
    }
}

int Harris::cornerDetection(float *ROfImage, int *isCorner, int width, int height)
{
    if (ROfImage == NULL || isCorner == NULL)
        return NULL_POINTER;

    int errcode;
    // 为R值申请设备空间并拷贝到设备
    float *d_ROfImage;
    int size1 = width * height * sizeof(float);
    cudaMalloc((void **)&d_ROfImage, size1);
    cudaMemcpy(d_ROfImage, ROfImage, size1, cudaMemcpyHostToDevice);

    // 为判断矩阵开辟设备空间
    int *d_isCorner;
    int size2 = width * height * sizeof(int);
    cudaMalloc((void **)&d_isCorner, size2);

    // 初始化非极大值抑制模板并将其拷贝到设备
    Template *tl;
    errcode = TemplateBasicOp::newTemplate(&tl);
    if (errcode != NO_ERROR) {
        cudaFree(d_ROfImage);
        cudaFree(d_isCorner);
        return errcode;
    }
    errcode = initialTemplate(tl, maxDimension, NULL);
    if (errcode != NO_ERROR) {
        cudaFree(d_ROfImage);
        cudaFree(d_isCorner);
        TemplateBasicOp::deleteTemplate(tl);
        return errcode;
    }
    TemplateCuda *tlCuda = TEMPLATE_CUDA(tl);
    errcode = TemplateBasicOp::copyToCurrentDevice(tl);
    if (errcode != NO_ERROR) {
        cudaFree(d_ROfImage);
        cudaFree(d_isCorner);
        TemplateBasicOp::deleteTemplate(tl);
        return errcode;
    }

    dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height + blocksize.y - 1) / blocksize.y;

    _cornerDetection<<<gridsize,blocksize>>>(d_ROfImage, d_isCorner, *tlCuda, width, height, this->threshold);

    cudaMemcpy(isCorner, d_isCorner, size2, cudaMemcpyDeviceToHost);

    cudaFree(d_ROfImage);
    cudaFree(d_isCorner);
    TemplateBasicOp::deleteTemplate(tl);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return errcode;
}


//************************************************************************************************************

//************************************************************************************************************

// 当下面函数运行出错时，使用该宏清除内存，防止内存泄漏。
#define CALHARRIS_FREE  do {                               \
        if (changeX != NULL) {                             \
            TemplateBasicOp::deleteTemplate(changeX);      \
            changeX = NULL;                                \
        }                                                  \
                                                           \
        if (changeY != NULL) {                             \
            TemplateBasicOp::deleteTemplate(changeY);      \
            changeY = NULL;                                \
        }                                                  \
                                                           \
        if (gaussianTl != NULL) {                          \
            TemplateBasicOp::deleteTemplate(gaussianTl);   \
            gaussianTl = NULL;                             \
        }                                                  \
                                                           \
        if (tl != NULL) {                                  \
            TemplateBasicOp::deleteTemplate(tl);           \
            tl = NULL;                                     \
        }                                                  \
                                                           \
        if (d_isCorner != NULL) {                          \
            delete []d_isCorner;                           \
            d_isCorner = NULL;                             \
        }                                                  \
                                                           \
        if (d_changeMatrix != NULL) {                      \
            delete []d_changeMatrix;                       \
            d_changeMatrix = NULL;                         \
        }                                                  \
                                                           \
        if (d_ROfImage != NULL) {                          \
            delete []d_ROfImage;                           \
            d_ROfImage = NULL;                             \
        }                                                  \
                                                           \
} while(0)                                                 \

int Harris::calHarris(Image *inImage, int *isCorner)
{
    if (inImage == NULL || isCorner == NULL)
        return NULL_POINTER;

    int errcode;

    Template *changeX;
    Template *changeY;
    Template *gaussianTl;
    Template *tl;

    int *d_isCorner;
    TwoDMatrix *d_changeMatrix;
    float *d_ROfImage;

    // 水平差分算子
    float dx[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    // 竖直差分算子
    float dy[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};

    float *gaussianValue = new float[gaussianDimension * gaussianDimension];
    errcode = initialGaussian(gaussianDimension, gaussianValue);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }

    errcode = TemplateBasicOp::newTemplate(&changeX);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }
    errcode = TemplateBasicOp::newTemplate(&changeY);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }
    errcode = TemplateBasicOp::newTemplate(&gaussianTl);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }
    errcode = TemplateBasicOp::newTemplate(&tl);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }

    errcode = initialTemplate(changeX, 3, dx);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }
    errcode = initialTemplate(changeY, 3, dy);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }
    errcode = initialTemplate(gaussianTl, gaussianDimension, gaussianValue);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }
    errcode = initialTemplate(tl, maxDimension, NULL);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }
    // 获取模板的cuda类型数据
    TemplateCuda *changeXCuda = TEMPLATE_CUDA(changeX);
    TemplateCuda *changeYCuda = TEMPLATE_CUDA(changeY);
    TemplateCuda *gaussianTlCuda = TEMPLATE_CUDA(gaussianTl);
    TemplateCuda *tlCuda = TEMPLATE_CUDA(tl);

    errcode = TemplateBasicOp::copyToCurrentDevice(changeX);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }
    errcode = TemplateBasicOp::copyToCurrentDevice(changeY);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }
    errcode = TemplateBasicOp::copyToCurrentDevice(gaussianTl);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }
    errcode = TemplateBasicOp::copyToCurrentDevice(tl);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }

    errcode = ImageBasicOp::copyToCurrentDevice(inImage);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inImage, &insubimgCud);
    if (errcode != NO_ERROR) {
        CALHARRIS_FREE;
        return errcode;
    }

    int size1 = inImage->width * inImage->height * sizeof(int);
    cudaMalloc((void **)&d_isCorner, size1);

    int size2 = inImage->width * inImage->height * sizeof(TwoDMatrix);
    cudaMalloc((void **)&d_changeMatrix, size2);

    int size3 = inImage->width * inImage->height * sizeof(float);
    cudaMalloc((void **)&d_ROfImage, size3);

    dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (inImage->width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (inImage->height + blocksize.y - 1) / blocksize.y;

    _calChangeMatrix<<<gridsize,blocksize>>>(insubimgCud, d_changeMatrix, *changeXCuda, *changeYCuda);

    _gaussianChangeMatrix<<<gridsize,blocksize>>>(d_changeMatrix, d_ROfImage, *gaussianTlCuda, 
                                                 inImage->width, inImage->height, this->k, 
                                                 this->gaussianDimension);

    _cornerDetection<<<gridsize,blocksize>>>(d_ROfImage, d_isCorner, *tlCuda, inImage->width, 
                                             inImage->height, this->threshold);

    cudaMemcpy(isCorner, d_isCorner, size1, cudaMemcpyDeviceToHost);

    TemplateBasicOp::deleteTemplate(changeX);
    TemplateBasicOp::deleteTemplate(changeY);
    TemplateBasicOp::deleteTemplate(gaussianTl);
    TemplateBasicOp::deleteTemplate(tl);
    cudaFree(d_isCorner);
    cudaFree(d_changeMatrix);
    cudaFree(d_ROfImage);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return errcode;
}

//************************************************************************************************************

//************************************************************************************************************

int Harris::calChangeMatrixOnCPU(Image *inImage, TwoDMatrix *changeMatrix)
{
    if (inImage == NULL || changeMatrix == NULL)
        return NULL_POINTER;

    int errcode;

    Template *changeX;
    Template *changeY;
    // 水平差分算子
    float dx[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    // 竖直差分算子
    float dy[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    // 创建模板
    errcode = TemplateBasicOp::newTemplate(&changeX);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = TemplateBasicOp::newTemplate(&changeY);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        return errcode;
    }
    // 初始化模板
    errcode = initialTemplate(changeX, 3, dx);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        TemplateBasicOp::deleteTemplate(changeY);
        return errcode;
    }
    errcode = initialTemplate(changeY, 3, dy);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        TemplateBasicOp::deleteTemplate(changeY);
        return errcode;
    }

    // 获取模板的cuda类型数据
    TemplateCuda *changeXCuda = TEMPLATE_CUDA(changeX);
    TemplateCuda *changeYCuda = TEMPLATE_CUDA(changeY);

    ImageBasicOp::copyToHost(inImage);

    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inImage, &insubimgCud);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(changeX);
        TemplateBasicOp::deleteTemplate(changeY);
        return errcode;
    }

    int index;
    // 初始化Ix，Iy
    float Ix = 0;
    float Iy = 0;
    int cx, cy;

    for (int r = 0;r < insubimgCud.imgMeta.height; r++) {
        for (int c = 0;c < insubimgCud.imgMeta.width; c++) {

            index = r * insubimgCud.imgMeta.width + c;
            Ix = 0;
            Iy = 0;

            // 舍弃边界,使边界元素的变换矩阵值为0（由于模板大小为3，所以舍弃第一行、第一列、最后一行、最后一列）
            if (c == 0 || c == insubimgCud.imgMeta.width - 1 || r == 0 || r == insubimgCud.imgMeta.height - 1) {
                changeMatrix[index].x1 = 0;
                changeMatrix[index].x2 = 0;
                changeMatrix[index].x3 = 0;
                changeMatrix[index].x4 = 0;
                continue;
            }

            // 通过模板计算Ix
            for(int i = 0; i < changeXCuda->tplMeta.count; i++) {
                cx = c + changeXCuda->tplMeta.tplData[2 * i];
                cy = r + changeXCuda->tplMeta.tplData[2 * i + 1];
                Ix += ((float)insubimgCud.imgMeta.imgData[cy * insubimgCud.pitchBytes + cx] * changeXCuda->attachedData[i]);
            }

            // 通过模板计算Iy
            for(int i = 0; i < changeYCuda->tplMeta.count; i++) {
                cx = c + changeYCuda->tplMeta.tplData[2 * i];
                cy = r + changeYCuda->tplMeta.tplData[2 * i + 1];
                //Iy += ((float)inimg.imgMeta.imgData[cy * inimg.pitchBytes + cx] * ATTACHED_DATA(&tply)[i]);
                //ATTACHED_DATA(&tply)该函数有问题
        
                Iy += ((float)insubimgCud.imgMeta.imgData[cy * insubimgCud.pitchBytes + cx] * changeYCuda->attachedData[i]);
            }

            // 计算变换矩阵
            changeMatrix[index].x1 = Ix * Ix;
            changeMatrix[index].x2 = Ix * Iy;
            changeMatrix[index].x3 = Ix * Iy;
            changeMatrix[index].x4 = Iy * Iy;
        }
    }

    TemplateBasicOp::deleteTemplate(changeX);
    TemplateBasicOp::deleteTemplate(changeY);

    return errcode;
}

//************************************************************************************************************

//************************************************************************************************************

int Harris::gaussianChangeMatrixOnCPU(TwoDMatrix *changeMatrix, float *ROfImage, int width, int height)
{
    if (changeMatrix == NULL || ROfImage == NULL)
        return NULL_POINTER;

    int errcode;
    // 高斯模板附加数据数组
    float *gaussianValue = new float[gaussianDimension * gaussianDimension];
    errcode = initialGaussian(gaussianDimension, gaussianValue);
    if (errcode != NO_ERROR) {
        delete []gaussianValue;
        return errcode;
    }

    // 初始化高斯模板
    Template *gaussianTl;
    errcode = TemplateBasicOp::newTemplate(&gaussianTl);
    if (errcode != NO_ERROR) {
        delete []gaussianValue;
        return errcode;
    }

    errcode = initialTemplate(gaussianTl, gaussianDimension, gaussianValue);
    if (errcode != NO_ERROR) {
        delete []gaussianValue;
        TemplateBasicOp::deleteTemplate(gaussianTl);
        return errcode;
    }

    TemplateCuda *gaussianTlCuda = TEMPLATE_CUDA(gaussianTl);

    int index;
    int ignore;
    // 变换矩阵局部变量
    TwoDMatrix temp;
    int cx, cy;

    for (int r = 0;r < height; r++) {
        for (int c = 0;c < width; c++) {

            index = r * width + c;

            // 舍弃边缘
            ignore = (gaussianDimension + 1) / 2; 
            if (c < ignore || c >= width - ignore || r < ignore || r >= height - ignore) {
                ROfImage[index] = 0;
                continue;
            }
            
            temp.x1 = 0;
            temp.x2 = 0;
            temp.x3 = 0;
            temp.x4 = 0;
            
            // 高斯平滑
            for(int i = 0; i < gaussianTlCuda->tplMeta.count; i++) {
                cx = c + gaussianTlCuda->tplMeta.tplData[2 * i];
                cy = r + gaussianTlCuda->tplMeta.tplData[2 * i + 1];
                temp.x1 += (changeMatrix[cy * width + cx].x1 * gaussianTlCuda->attachedData[i]);
                temp.x2 += (changeMatrix[cy * width + cx].x2 * gaussianTlCuda->attachedData[i]);
                temp.x3 += (changeMatrix[cy * width + cx].x3 * gaussianTlCuda->attachedData[i]);
                temp.x4 += (changeMatrix[cy * width + cx].x4 * gaussianTlCuda->attachedData[i]);
            }

            // 计算该像素点的R值
            ROfImage[index] = temp.x1 * temp.x4 - temp.x2 * temp.x3 - k * (temp.x1 + temp.x4) * (temp.x1 + temp.x4);
        }
    }

    delete []gaussianValue;
    TemplateBasicOp::deleteTemplate(gaussianTl);

    return errcode;
}

//************************************************************************************************************

//************************************************************************************************************

int Harris::cornerDetectionOnCPU(float *ROfImage, int *isCorner, int width, int height)
{
    if (ROfImage == NULL || isCorner == NULL)
        return NULL_POINTER;

    int errcode;

    // 初始化非极大值抑制模板并将其拷贝到设备
    Template *tl;
    errcode = TemplateBasicOp::newTemplate(&tl);
    if (errcode != NO_ERROR) {
        return errcode;
    }
    errcode = initialTemplate(tl, maxDimension, NULL);
    if (errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(tl);
        return errcode;
    }

    TemplateCuda *tlCuda = TEMPLATE_CUDA(tl);

    int index;
    float max = 0;
    int cx, cy;

    for (int r = 0;r < height; r++) {
        for (int c = 0;c < width; c++) {

            index = r * width + c;

            isCorner[index] = 0;
            max = 0;
            
            // 判断是否为角点
            if (ROfImage[index] > threshold) {
                for(int i = 0; i < tlCuda->tplMeta.count; i++) {
                    cx = c + tlCuda->tplMeta.tplData[2 * i];
                    cy = r + tlCuda->tplMeta.tplData[2 * i + 1];
                    if ((cx < width) && (cx >= 0) && (cy < height) && (cy >= 0)) {
                        max = (ROfImage[cy * width + cx] > max) ? ROfImage[cy * width + cx] : max;
                    }
                }
                if (max == ROfImage[index]) {
                    isCorner[index] = 1;
                }
                // 非极大值抑制
                else {
                    isCorner[index] = 0;
                }
            }
            else {
                isCorner[index] = 0;
            }
        }
    }
    TemplateBasicOp::deleteTemplate(tl);

    return errcode;
}

//************************************************************************************************************

//************************************************************************************************************

int Harris::calHarrisOnCPU(Image *inImage, int *isCorner)
{
    int size = inImage->width * inImage->height;
    TwoDMatrix *changeMatrix = new TwoDMatrix[size];
    float *ROfImage = new float[size];

    calChangeMatrixOnCPU(inImage, changeMatrix);
    gaussianChangeMatrixOnCPU(changeMatrix, ROfImage, inImage->width, inImage->height);
    cornerDetectionOnCPU(ROfImage, isCorner, inImage->width, inImage->height);

    delete []changeMatrix;
    delete []ROfImage;

    return 0;
}