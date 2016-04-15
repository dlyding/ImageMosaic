#include <iostream>
#include <cmath>
#include "SomeFunction.h"
#include "CovarianceMatrix.h"
#include "ErrorCode.h"
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 浮点数转整形的宏定义
#define FloatToInteger(fNum) ((fNum > 0) ? static_cast<int>(fNum + 0.5) : static_cast<int>(fNum - 0.5))

// 计算一点变换后的坐标的宏定义
#define calz(matrix, x, y) ((x) * matrix[6] + (y) * matrix[7] + matrix[8])
#define calx(matrix, x, y) ((((x) * matrix[0] + (y) * matrix[1] + matrix[2])) / (calz(matrix, x, y)))
#define caly(matrix, x, y) ((((x) * matrix[3] + (y) * matrix[4] + matrix[5])) / (calz(matrix, x, y)))


// 纹理内存只能用于全局变量，使用全局存储时需要加入边界判断，经测试效率不及
// 纹理内存，纹理拾取返回的数据类型 unsigned char 型，维度为2，返回类型不转换
static texture<unsigned char, 2, cudaReadModeElementType> _bilateralInimgTex;

__global__ void _imageConnect(
        ImageCuda inimg1,      // 输入图像1
        ImageCuda inimg2,      // 输入图像2
        ImageCuda outimg       // 输出图像
);

__global__ void _imageConnect(ImageCuda inimg1, ImageCuda inimg2, ImageCuda outimg)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;

	if (c >= outimg.imgMeta.width || r >= outimg.imgMeta.height)
    	return;

    int index = r * outimg.pitchBytes + c;
    // 图像1像素点索引
    int index1;
    // 图像2像素点索引
    int index2;

    outimg.imgMeta.imgData[index] = 0;

    // 判断像素点在哪一幅图像上
    if (c < inimg1.imgMeta.width) {
        // 判断像素点是否越界
        if (r < inimg1.imgMeta.height) {
            index1 = r * inimg1.pitchBytes + c;
            outimg.imgMeta.imgData[index] = inimg1.imgMeta.imgData[index1];
        }
        /*else {
            outimg.imgMeta.imgData[index] = 0;
        }*/
    }
    else {
        // 判断像素点是否越界
        if (r < inimg2.imgMeta.height) {
            index2 = r * inimg2.pitchBytes + c - inimg1.imgMeta.width;
            outimg.imgMeta.imgData[index] = inimg2.imgMeta.imgData[index2];
        }
        /*else {
            outimg.imgMeta.imgData[index] = 0;
        }*/
    }
}

int imageConnect(Image *inImg1, Image *inImg2, Image *outImg)
{
	if (inImg1 == NULL || inImg2 == NULL || outImg == NULL)
		return NULL_POINTER;

    int errcode;

    // 输出图像高度为输入的两幅图像高度的较大者
	int height = (inImg1->height > inImg2->height) ? inImg1->height : inImg2->height;
    // 输出图像宽度为输入的两幅图像宽度的和
	int width = inImg1->width + inImg2->width;
    // 为输出图像申请空间
	errcode = ImageBasicOp::makeAtHost(outImg, width, height);
    if (errcode != NO_ERROR)
        return errcode;

    // 将图像1拷贝到设备并提取子图像
    // 先拷贝再提取子图像
    errcode = ImageBasicOp::copyToCurrentDevice(inImg1);
    if (errcode != NO_ERROR)
        return errcode;
	ImageCuda inimgcud1;
	errcode = ImageBasicOp::roiSubImage(inImg1, &inimgcud1);
	if (errcode != NO_ERROR)
        return errcode;

    // 将图像2拷贝到设备并提取子图像
    errcode = ImageBasicOp::copyToCurrentDevice(inImg2);
    if (errcode != NO_ERROR)
        return errcode;
	ImageCuda inimgcud2;
	errcode = ImageBasicOp::roiSubImage(inImg2, &inimgcud2);
    if (errcode != NO_ERROR)
        return errcode;
	
    // 将输出图像拷贝到设备并提取子图像
    errcode = ImageBasicOp::copyToCurrentDevice(outImg);
    if (errcode != NO_ERROR)
        return errcode;
	ImageCuda outimgcud;
	errcode = ImageBasicOp::roiSubImage(outImg, &outimgcud);
    if (errcode != NO_ERROR)
        return errcode;
	

	dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outimgcud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outimgcud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    _imageConnect<<<gridsize,blocksize>>>(inimgcud1, inimgcud2, outimgcud);

    ImageBasicOp::copyToHost(outImg);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return errcode;
}

//*************************************************************************************

//*************************************************************************************


int drawLine(Image *inImg, int x1, int y1, int x2, int y2, int pixel)
{
    int dx = x2 - x1;
    int dy = y2 - y1;
    // 求出x方向和y方向的变化最大值
    int maxstep = (abs(dx) > abs(dy)) ? abs(dx) : abs(dy);
    float xstep;
    float ystep;

    // x方向的步长
    xstep = static_cast<float>(dx) / static_cast<float>(maxstep);
    // y方向的步长
    ystep = static_cast<float>(dy) / static_cast<float>(maxstep);

    inImg->imgData[y1 * inImg->width + x1] = pixel;

    float x = static_cast<float>(x1);
    float y = static_cast<float>(y1);

    // 按x方向和y方向步长依次画出直线
    for (int i = 0; i <= maxstep; i++) {
        x += xstep;
        y += ystep;
        inImg->imgData[FloatToInteger(y) * inImg->width + FloatToInteger(x)] = pixel;
    }

    return 0;
}


//*************************************************************************************************

//*************************************************************************************************

static __host__ int      // 返回值：若正确执行返回 NO_ERROR
_initTexture(
        Image *inimg  // 输入图像
);

__global__ void _imageTransform(
        ImageCuda inimg,                      // 输入图像
        ImageCuda outimg,                     // 变换后的图像
        float *d_inverseTransformMatrix,      // 变换矩阵
        float distancex,                      // 水平平移距离
        float distancey                       // 竖直平移距离
);

// Host 函数：initTexture（初始化纹理内存）
static __host__ int _initTexture(Image *inimg)
{
    cudaError_t cuerrcode;
    int errcode;  // 局部变量，错误码

    // 将输入图像拷贝到 Device 内存中。
    errcode = ImageBasicOp::copyToCurrentDevice(inimg);
    if (errcode != NO_ERROR)
        return errcode;
        
    // 提取输入图像的 ROI 子图像。
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inimg, &insubimgCud);
    if (errcode != NO_ERROR)
        return errcode;

    // 设置数据通道描述符，因为只有一个颜色通道（灰度图），因此描述符中只有第一
    // 个分量含有数据。概述据通道描述符用于纹理内存的绑定操作。
    struct cudaChannelFormatDesc chndesc;
    chndesc = cudaCreateChannelDesc(sizeof (unsigned char) * 8, 0, 0, 0,
                                    cudaChannelFormatKindUnsigned);

    /*_bilateralInimgTex.addressMode[0] = cudaAddressModeWrap;
    _bilateralInimgTex.addressMode[1] = cudaAddressModeWrap;
    _bilateralInimgTex.filterMode = cudaFilterModeLinear;
    _bilateralInimgTex.normalized = 0;
*/              //为什么加这个就不对？？？？？？？？？？？？？？？？？？？？？？
    // 将输入图像数据绑定至纹理内存（texture） 
    cuerrcode = cudaBindTexture2D(
            NULL, &_bilateralInimgTex, insubimgCud.imgMeta.imgData, &chndesc, 
            insubimgCud.imgMeta.width, insubimgCud.imgMeta.height, 
            insubimgCud.pitchBytes);

    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;
    return NO_ERROR;
}

__global__ void _imageTransform(ImageCuda inimg, ImageCuda outimg, float *d_inverseTransformMatrix,
                                float distancex, float distancey)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= outimg.imgMeta.width || r >= outimg.imgMeta.height)
        return;

    int index = r * outimg.pitchBytes + c;
    outimg.imgMeta.imgData[index] = 0;

    float x = calx(d_inverseTransformMatrix, c - distancex, r - distancey);
    float y = caly(d_inverseTransformMatrix, c - distancex, r - distancey);

/*    float x = c;
    float y = r;*/

    if(FloatToInteger(x) < 0 || FloatToInteger(x) >= inimg.imgMeta.width || 
        FloatToInteger(y) < 0 || FloatToInteger(y) >= inimg.imgMeta.height)
        return;

    outimg.imgMeta.imgData[index] = tex2D(_bilateralInimgTex, x, y);
}

int imageTransform(Image *inImg, Image *outImg, float *transformMatrix, int matrixSize, 
                   float &distancex, float &distancey)
{
    if (inImg == NULL || outImg == NULL || transformMatrix == NULL)
        return NULL_POINTER;

    int errcode;

    // 为变换矩阵的逆矩阵开辟空间
    float *inverseTransformMatrix = new float[matrixSize];
    // 为变换矩阵的逆矩阵开辟设备空间
    float *d_inverseTransformMatrix;
    cudaMalloc((void **)&d_inverseTransformMatrix, matrixSize * sizeof(float));

    // 计算变换矩阵的逆矩阵
    calInverseMatrix(transformMatrix, inverseTransformMatrix, 3, 3);
/*
    for(int i = 0; i < matrixSize; i++) {
        cout<<inverseTransformMatrix[i]<<endl;
    }*/

    cudaMemcpy(d_inverseTransformMatrix, inverseTransformMatrix, matrixSize * sizeof(float), 
               cudaMemcpyHostToDevice);

    // 初始化纹理内存
    _initTexture(inImg);

    float x, y, minx, maxx, miny, maxy; // 平移距离
    
    // 通过计算图像的四个顶点变换后的坐标确定变换后的图像大小
    // 如果变换后图像坐标存在负值，则将变换后的图像向右、向下平移若干单位，
    // 并记录在distancex和distancey中。下一步图像融合时，也需将另一幅输入图像
    // 向右、向下平移对应若干单位
    x = calx(transformMatrix, 0, 0);
    y = caly(transformMatrix, 0, 0);

    minx = maxx = x;
    miny = maxy = y;

    x = calx(transformMatrix, inImg->width - 1, 0);
    y = caly(transformMatrix, inImg->width - 1, 0);

    minx = (x < minx) ? x : minx;
    maxx = (x > maxx) ? x : maxx;
    miny = (y < miny) ? y : miny;
    maxy = (y > maxy) ? y : maxy;

    x = calx(transformMatrix, 0, inImg->height - 1);
    y = caly(transformMatrix, 0, inImg->height - 1);

    minx = (x < minx) ? x : minx;
    maxx = (x > maxx) ? x : maxx;
    miny = (y < miny) ? y : miny;
    maxy = (y > maxy) ? y : maxy;

    x = calx(transformMatrix, inImg->width - 1, inImg->height - 1);
    y = caly(transformMatrix, inImg->width - 1, inImg->height - 1);

    minx = (x < minx) ? x : minx;
    maxx = (x > maxx) ? x : maxx;
    miny = (y < miny) ? y : miny;
    maxy = (y > maxy) ? y : maxy;

    /*cout<<minx<<endl;
    cout<<maxx<<endl;
    cout<<miny<<endl;
    cout<<maxy<<endl<<endl;
    cout<<inImg->width<<endl;
    cout<<inImg->height<<endl<<endl;*/

    distancex = (minx < 0) ? -minx : 0;
    distancey = (miny < 0) ? -miny : 0;

    // 计算输出图像宽度和高度
    int width = FloatToInteger(maxx + distancex) + 1; // 图像只能向右移、下移，不能向左移、上移
    int height = FloatToInteger(maxy + distancey) + 1;
    // 为输出图像开辟空间
    errcode = ImageBasicOp::makeAtHost(outImg, width, height);
    if (errcode != NO_ERROR) {
        delete []inverseTransformMatrix;
        cudaFree(d_inverseTransformMatrix);
        return errcode;
    }

    /*cout<<outImg->width<<endl;
    cout<<outImg->height<<endl;*/

    // 将输入图像拷贝到设备并提取子图像
    errcode = ImageBasicOp::copyToCurrentDevice(inImg);
    if (errcode != NO_ERROR) {
        delete []inverseTransformMatrix;
        cudaFree(d_inverseTransformMatrix);
        return errcode;
    }
    ImageCuda inimgcud;
    errcode = ImageBasicOp::roiSubImage(inImg, &inimgcud);
    if (errcode != NO_ERROR) {
        delete []inverseTransformMatrix;
        cudaFree(d_inverseTransformMatrix);
        return errcode;
    }

    // 将输出图像拷贝到设备并提取子图像
    errcode = ImageBasicOp::copyToCurrentDevice(outImg);
    if (errcode != NO_ERROR) {
        delete []inverseTransformMatrix;
        cudaFree(d_inverseTransformMatrix);
        return errcode;
    }
    ImageCuda outimgcud;
    errcode = ImageBasicOp::roiSubImage(outImg, &outimgcud);
    if (errcode != NO_ERROR) {
        delete []inverseTransformMatrix;
        cudaFree(d_inverseTransformMatrix);
        return errcode;
    }

    dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outimgcud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outimgcud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    _imageTransform<<<gridsize, blocksize>>>(inimgcud, outimgcud, d_inverseTransformMatrix, 
                                             distancex, distancey);

    ImageBasicOp::copyToHost(outImg);

    delete []inverseTransformMatrix;
    cudaFree(d_inverseTransformMatrix);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return errcode;
}

//**************************************************************************************************************

//**************************************************************************************************************

int imageTranslation(Image *inImg, Image *outImg, float distancex, float distancey)
{
    int errcode;
    // 初始化变换矩阵，由于只是针对平移，所以只有第三位和第六位被赋值
    float transformMatrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    transformMatrix[2] = distancex;
    transformMatrix[5] = distancey;
    errcode = imageTransform(inImg, outImg, transformMatrix, 9, distancex, distancey);
    if (errcode != NO_ERROR)
        return errcode;
    return errcode;
}

//**************************************************************************************************************

//**************************************************************************************************************

__global__ void _imageFusion(
        ImageCuda inimg1,       // 输入图像1
        ImageCuda inimg2,       // 输入图像2
        ImageCuda outimg        // 输出图像
);

__global__ void _imageFusion(ImageCuda inimg1, ImageCuda inimg2, ImageCuda outimg)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= outimg.imgMeta.width || r >= outimg.imgMeta.height)
        return;

    int index = r * outimg.pitchBytes + c;
    int index1 = r * inimg1.pitchBytes + c;
    int index2 = r * inimg2.pitchBytes + c;

    // 像素点属于图像2，不属于图像1时，输出图像取图像2像素值
    if ((c >= inimg1.imgMeta.width || r >= inimg1.imgMeta.height) 
        && (c < inimg2.imgMeta.width && r < inimg2.imgMeta.height)) {
        outimg.imgMeta.imgData[index] = inimg2.imgMeta.imgData[index2];
    }

    // 像素点属于图像1，不属于图像2时，输出图像取图像1像素值
    else if ((c >= inimg2.imgMeta.width || r >= inimg2.imgMeta.height) 
             && (c < inimg1.imgMeta.width && r < inimg1.imgMeta.height)) {
        outimg.imgMeta.imgData[index] = inimg1.imgMeta.imgData[index1];
    }

    // 像素点同时属于图像1和图像2时，输出图像取图像1和图像2像素值和的平均值
    else if ((c < inimg2.imgMeta.width && r < inimg2.imgMeta.height) 
             && (c < inimg1.imgMeta.width && r < inimg1.imgMeta.height)) {
        if (inimg1.imgMeta.imgData[index1] == 0) {
            outimg.imgMeta.imgData[index] = inimg2.imgMeta.imgData[index2];
        }

        else if (inimg2.imgMeta.imgData[index2] == 0) {
            outimg.imgMeta.imgData[index] = inimg1.imgMeta.imgData[index1];
        }

        else {
            outimg.imgMeta.imgData[index] = (inimg1.imgMeta.imgData[index1] + 
                                             inimg2.imgMeta.imgData[index2]) / 2;
        }
    }

    // 像素点都不属于图像1和图像2时，输出图像取0
    else {
        outimg.imgMeta.imgData[index] = 0;
        
    }
}

int imageFusion(Image *inImg1, Image *inImg2, Image *outImg)
{
    if (inImg1 == NULL || inImg2 == NULL || outImg == NULL)
        return NULL_POINTER;

    int errcode;
    // 输出图像高度取输入图像高度较大者
    int height = (inImg1->height > inImg2->height) ? inImg1->height : inImg2->height;
    // 输出图像宽度取输入图像宽度较大者
    int width = (inImg1->width > inImg2->width) ? inImg1->width : inImg2->width;
    // 为输出图像申请空间
    errcode = ImageBasicOp::makeAtHost(outImg, width, height);
    if (errcode != NO_ERROR) 
        return errcode;

    // 将图像1拷贝到设备并提取子图像
    // 先拷贝再提取子图像
    errcode = ImageBasicOp::copyToCurrentDevice(inImg1);
    if (errcode != NO_ERROR) 
        return errcode;
    ImageCuda inimgcud1;
    errcode = ImageBasicOp::roiSubImage(inImg1, &inimgcud1);
    if (errcode != NO_ERROR) 
        return errcode;
    
    // 将图像2拷贝到设备并提取子图像
    errcode = ImageBasicOp::copyToCurrentDevice(inImg2);
    if (errcode != NO_ERROR) 
        return errcode;
    ImageCuda inimgcud2;
    errcode = ImageBasicOp::roiSubImage(inImg2, &inimgcud2);
    if (errcode != NO_ERROR) 
        return errcode;
    
    // 将输出图像拷贝到设备并提取子图像
    errcode = ImageBasicOp::copyToCurrentDevice(outImg);
    if (errcode != NO_ERROR) 
        return errcode;
    ImageCuda outimgcud;
    errcode = ImageBasicOp::roiSubImage(outImg, &outimgcud);
    if (errcode != NO_ERROR) 
        return errcode;

    dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outimgcud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outimgcud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    _imageFusion<<<gridsize,blocksize>>>(inimgcud1, inimgcud2, outimgcud);

    ImageBasicOp::copyToHost(outImg);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return 0;
}