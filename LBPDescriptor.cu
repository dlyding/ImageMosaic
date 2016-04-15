#include <iostream>
#include <cstring>
#include "LBPDescriptor.h"
#include "ErrorCode.h"
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

// 按如下图方式初始化LBP模板
// |      11,12,13     |
// |   10,         14  |
// |25,   1, 2, 3,   15|
// |24,   8, 9, 4,   16|
// |23,   7, 6, 5,   17|
// |   22,         18  |
// |      21,20,19     |
int initialLBPTemplate(
        Template *LBPTl   // LBP模板，传入前需要new
);

// 计算数组元素所能组成的最小十进制数
__device__ int calArrayMin(
        int *inArray,     // 输入数组
        int num           // 数组大小
);

// 数组右移一位
__device__ void arrayRightShift(
        int *inArray,     // 输入数组
        int num           // 数组大小
);

__global__ void _calLBPDescriptor(
        ImageCuda inimg,                    // 输入图像
        int *d_isCorner,                    // 判断矩阵
        Descriptor *d_descriptorOfCorner,   // 描述符矩阵
        Template tpl                        // LBP模板
);

__device__ int calArrayMin(int *inArray, int num)
{
	int min = 65536;
	int sum = 0;
	for (int i = 0; i < num; i++) {
        // 二进制转换为十进制
		for(int j = 0; j < num; j++) {
			sum *= 2;
			sum += inArray[j];
		}
		min = (sum < min) ? sum : min;
		arrayRightShift(inArray, num);
        sum = 0;
	}
    return min;
}

__device__ void arrayRightShift(int *inArray, int num)
{
	int temp;
	temp = inArray[num - 1];
	for (int i = num - 2; i >= 0; i--) {
		inArray[i + 1] = inArray[i];
	}
	inArray[0] = temp;
}

int initialLBPTemplate(Template *LBPTl)
{
    if (LBPTl == NULL)
        return NULL_POINTER;
    int errcode;
	errcode = TemplateBasicOp::makeAtHost(LBPTl, 5 * 5);
    if (errcode != NO_ERROR)
        return errcode;
	int tldata[] = {-1,-1, 0,-1, 1,-1, 1,0, 1,1, 0,1, -1,1, -1,0, 0,0, -2,-2, -1,-3, 0,-3, 1,-3, 
				    2,-2, 3,-1, 3,0, 3,1, 2,2, 1,3, 0,3, -1,3, -2,2, -3,1, -3,0, -3,-1};
	memcpy(LBPTl->tplData, tldata, 50 * sizeof(int));
/*    for(int i = 0; i < 50; i++) {
        cout<<LBPTl->tplData[i]<<" ";
    }*/

    return errcode;
}

__global__ void _calLBPDescriptor(ImageCuda inimg, int *d_isCorner, 
                                  Descriptor *d_descriptorOfCorner, Template tpl)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;

	if (c >= inimg.imgMeta.width || r >= inimg.imgMeta.height)
    	return;

    int index = r * inimg.imgMeta.width + c;     

    int indexofinimg = r * inimg.pitchBytes + c;   // 图像的宽度和矩阵的宽度不一样

    // 舍弃边缘
    if (c <= 2 || c >= inimg.imgMeta.width - 3 || r <= 2 || r >= inimg.imgMeta.height - 3) {
    	d_descriptorOfCorner[index].inner = 0;
    	d_descriptorOfCorner[index].outer = 0;
    	return;
    }

    int cx, cy;
    int d_inner[8];
    int d_outer[16];

    if (d_isCorner[index] == 1) {
        // 提取内层循环
    	for(int i = 0; i < 8; i++) {
    		cx = c + tpl.tplData[2 * i];
        	cy = r + tpl.tplData[2 * i + 1];
        	if (inimg.imgMeta.imgData[cy * inimg.pitchBytes + cx] > inimg.imgMeta.imgData[indexofinimg]) {
        		d_inner[i] = 1;
        	}
        	else {
        		d_inner[i] = 0;
        	}
    	}
        // 提取外层循环
    	for(int i = 9; i < 25; i++) {
    		cx = c + tpl.tplData[2 * i];
        	cy = r + tpl.tplData[2 * i + 1];
        	if (inimg.imgMeta.imgData[cy * inimg.pitchBytes + cx] > inimg.imgMeta.imgData[indexofinimg]) {
        		d_outer[i - 9] = 1;
        	}
        	else {
        		d_outer[i - 9] = 0;
        	}
    	}
    	d_descriptorOfCorner[index].inner = calArrayMin(d_inner, 8);
    	d_descriptorOfCorner[index].outer = calArrayMin(d_outer, 16);
    }

    else {
    	d_descriptorOfCorner[index].inner = 0;
    	d_descriptorOfCorner[index].outer = 0;
    }

}

int LBPDescriptor::calLBPDescriptor(Image *inImage, int *isCorner, Descriptor *descriptorOfCorner)
{
	if (inImage == NULL || isCorner == NULL || descriptorOfCorner == NULL)
		return NULL_POINTER;

    int errcode;

    // 初始化LBP模板并将其拷贝到设备
	Template *LBPTl;
	errcode = TemplateBasicOp::newTemplate(&LBPTl);
    if (errcode != NO_ERROR)
        return errcode;
	errcode = initialLBPTemplate(LBPTl);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }
	errcode = TemplateBasicOp::copyToCurrentDevice(LBPTl);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }

    // 将图像拷贝到设备并提取子图像
	ImageCuda insubimgCud;
	errcode = ImageBasicOp::copyToCurrentDevice(inImage);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }
	errcode = ImageBasicOp::roiSubImage(inImage, &insubimgCud);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }

    // 为判断矩阵申请设备空间并将其拷贝到设备
	int *d_isCorner;
	int size1 = inImage->width * inImage->height * sizeof(int);
	cudaMalloc((void **)&d_isCorner, size1);
	cudaMemcpy(d_isCorner, isCorner, size1, cudaMemcpyHostToDevice);

    // 为描述符矩阵申请设备空间
	Descriptor *d_descriptorOfCorner;
	int size2 = inImage->width * inImage->height * sizeof(Descriptor);
	cudaMalloc((void **)&d_descriptorOfCorner, size2);

	dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    _calLBPDescriptor<<<gridsize,blocksize>>>(insubimgCud, d_isCorner, d_descriptorOfCorner, *LBPTl);

    cudaMemcpy(descriptorOfCorner, d_descriptorOfCorner, size2, cudaMemcpyDeviceToHost);

    TemplateBasicOp::deleteTemplate(LBPTl);

    cudaFree(d_isCorner);
    cudaFree(d_descriptorOfCorner);

     // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return errcode;	
}

//***************************************************************************************************************

//***************************************************************************************************************

__global__ void _LBPMatch(
        Descriptor *d_descriptorOfCorner1,    // 输入图像1的描述符矩阵
        Descriptor *d_descriptorOfCorner2,    // 输入图像2的描述符矩阵
	    Coordinate *d_matchCorner,            // 匹配矩阵（如果（2,5）位置存放（3,8），则图像1（2,5）与
                                              // 图像2（3,8）匹配），传入前需要开辟空间
        int width1,                           // 输入图像1的宽度
        int height1,                          // 输入图像1的高度
        int width2,                           // 输入图像2的宽度
        int height2                           // 输入图像2的高度
);

__global__ void _LBPMatch(Descriptor *d_descriptorOfCorner1, Descriptor *d_descriptorOfCorner2, 
	                      Coordinate *d_matchCorner, int width1, int height1, int width2, int height2)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;

	if (c >= width1 || r >= height1)
    	return;

    int index1 = r * width1 + c;
    int index2;
    int innerdistance;
    int outerdistance;

    if (d_descriptorOfCorner1[index1].inner == 0) {
    	d_matchCorner[index1].x = 0;
    	d_matchCorner[index1].y = 0;
    }
    else {
    	for (int i = 0; i < height2; i++) {
    		for (int j = 0; j < width2; j++) {
    			index2 = i * width2 + j;
    			innerdistance = d_descriptorOfCorner1[index1].inner - d_descriptorOfCorner2[index2].inner;
    			innerdistance = (innerdistance > 0) ? innerdistance : -innerdistance;
    			outerdistance = d_descriptorOfCorner1[index1].outer - d_descriptorOfCorner2[index2].outer;
    			outerdistance = (outerdistance > 0) ? outerdistance : -outerdistance;
    			if (innerdistance < 1 && outerdistance < 1) {
    				d_matchCorner[index1].x = j;
    				d_matchCorner[index1].y = i;
                    return;
    			}  			
    		}
    	}
    	d_matchCorner[index1].x = 0;
    	d_matchCorner[index1].y = 0;
    }

}

//描述子不好，会多点对应到一点
int LBPDescriptor::LBPMatch(Descriptor *descriptorOfCorner1, Descriptor *descriptorOfCorner2, 
	                        Coordinate *matchCorner, int width1, int height1, int width2, int height2)
{
	if (descriptorOfCorner1 == NULL || descriptorOfCorner2 == NULL || matchCorner == NULL) 
		return NULL_POINTER;

    //int errcode;

    // 为图像1描述符矩阵申请设备空间并将其拷贝到设备
	Descriptor *d_descriptorOfCorner1;
	int size1 = width1 * height1 * sizeof(Descriptor);
	cudaMalloc((void **)&d_descriptorOfCorner1, size1);
	cudaMemcpy(d_descriptorOfCorner1, descriptorOfCorner1, size1, cudaMemcpyHostToDevice);

    // 为图像2描述符矩阵申请设备空间并将其拷贝到设备
	Descriptor *d_descriptorOfCorner2;
	int size2 = width2 * height2 * sizeof(Descriptor);
	cudaMalloc((void **)&d_descriptorOfCorner2, size2);
	cudaMemcpy(d_descriptorOfCorner2, descriptorOfCorner2, size2, cudaMemcpyHostToDevice);

    // 为匹配矩阵申请设备空间
	Coordinate *d_matchCorner;
	int size3 = width1 * height1 * sizeof(Coordinate);
	cudaMalloc((void **)&d_matchCorner, size3);

	dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (width1 + blocksize.x - 1) / blocksize.x;
    gridsize.y = (height1 + blocksize.y - 1) / blocksize.y;

    _LBPMatch<<<gridsize,blocksize>>>(d_descriptorOfCorner1, d_descriptorOfCorner2, d_matchCorner, 
    	                              width1, height1, width2, height2);

    cudaMemcpy(matchCorner, d_matchCorner, size3, cudaMemcpyDeviceToHost);

    cudaFree(d_descriptorOfCorner1);
    cudaFree(d_descriptorOfCorner2);
    cudaFree(d_matchCorner);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;

    return 0;
}

//*************************************************************************************************************

//*************************************************************************************************************

int LBPDescriptor::ImageMatchOfLBP(Image *inImage1, Image *inImage2, int *isCorner1, int *isCorner2, 
                    Coordinate *matchCorner)
{
    if (inImage1 == NULL || inImage2 == NULL || isCorner1 == NULL || isCorner2 == NULL || matchCorner == NULL)
        return NULL_POINTER;

    int errcode;
    // 初始化LBP模板并将其拷贝到设备
    Template *LBPTl;
    errcode = TemplateBasicOp::newTemplate(&LBPTl);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = initialLBPTemplate(LBPTl);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }
    errcode = TemplateBasicOp::copyToCurrentDevice(LBPTl);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }

    // 将图像1拷贝到设备并提取子图像
    ImageCuda insubimgCud1;
    errcode = ImageBasicOp::copyToCurrentDevice(inImage1);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }
    errcode = ImageBasicOp::roiSubImage(inImage1, &insubimgCud1);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }

    // 将图像2拷贝到设备并提取子图像
    ImageCuda insubimgCud2;
    errcode = ImageBasicOp::copyToCurrentDevice(inImage2);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }
    errcode = ImageBasicOp::roiSubImage(inImage2, &insubimgCud2);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }

    // 为判断矩阵1申请设备空间并将其拷贝到设备
    int *d_isCorner1;
    int size1 = inImage1->width * inImage1->height * sizeof(int);
    cudaMalloc((void **)&d_isCorner1, size1);
    cudaMemcpy(d_isCorner1, isCorner1, size1, cudaMemcpyHostToDevice);

    // 为判断矩阵2申请设备空间并将其拷贝到设备
    int *d_isCorner2;
    int size2 = inImage2->width * inImage2->height * sizeof(int);
    cudaMalloc((void **)&d_isCorner2, size2);
    cudaMemcpy(d_isCorner2, isCorner2, size2, cudaMemcpyHostToDevice);

    // 为描述符矩阵1申请设备空间
    Descriptor *d_descriptorOfCorner1;
    int size3 = inImage1->width * inImage1->height * sizeof(Descriptor);
    cudaMalloc((void **)&d_descriptorOfCorner1, size3);

    // 为描述符矩阵2申请设备空间
    Descriptor *d_descriptorOfCorner2;
    int size4 = inImage2->width * inImage2->height * sizeof(Descriptor);
    cudaMalloc((void **)&d_descriptorOfCorner2, size4);

    // 为匹配矩阵申请设备空间
    Coordinate *d_matchCorner;
    int size5 = inImage1->width * inImage1->height * sizeof(Coordinate);
    cudaMalloc((void **)&d_matchCorner, size5);

    dim3 blocksize,gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (insubimgCud1.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (insubimgCud1.imgMeta.height + blocksize.y - 1) / blocksize.y;

    _calLBPDescriptor<<<gridsize,blocksize>>>(insubimgCud1, d_isCorner1, d_descriptorOfCorner1, *LBPTl);

    _calLBPDescriptor<<<gridsize,blocksize>>>(insubimgCud2, d_isCorner2, d_descriptorOfCorner2, *LBPTl);

    _LBPMatch<<<gridsize,blocksize>>>(d_descriptorOfCorner1, d_descriptorOfCorner2, d_matchCorner, 
                                      inImage1->width, inImage1->height, inImage2->width, inImage2->height);

    cudaMemcpy(matchCorner, d_matchCorner, size5, cudaMemcpyDeviceToHost);

    TemplateBasicOp::deleteTemplate(LBPTl);

    cudaFree(d_isCorner1);
    cudaFree(d_isCorner2);
    cudaFree(d_descriptorOfCorner1);
    cudaFree(d_descriptorOfCorner2);
    cudaFree(d_matchCorner);

    // 若调用 CUDA 出错返回错误代码
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
    
    return errcode;
}

//*************************************************************************************************************

//*************************************************************************************************************

int calArrayMinOnCPU(
        int *inArray,     // 输入数组
        int num           // 数组大小
);

// 数组右移一位
void arrayRightShiftOnCPU(
        int *inArray,     // 输入数组
        int num           // 数组大小
);

int calArrayMinOnCPU(int *inArray, int num)
{
    int min = 65536;
    int sum = 0;
    for (int i = 0; i < num; i++) {
        // 二进制转换为十进制
        for(int j = 0; j < num; j++) {
            sum *= 2;
            sum += inArray[j];
        }
        min = (sum < min) ? sum : min;
        arrayRightShiftOnCPU(inArray, num);
        sum = 0;
    }
    return min;
}

void arrayRightShiftOnCPU(int *inArray, int num)
{
    int temp;
    temp = inArray[num - 1];
    for (int i = num - 2; i >= 0; i--) {
        inArray[i + 1] = inArray[i];
    }
    inArray[0] = temp;
}

int LBPDescriptor::calLBPDescriptorOnCPU(Image *inImage, int *isCorner, Descriptor *descriptorOfCorner)
{
    if (inImage == NULL || isCorner == NULL || descriptorOfCorner == NULL)
        return NULL_POINTER;

    int errcode;

    // 初始化LBP模板
    Template *LBPTl;
    errcode = TemplateBasicOp::newTemplate(&LBPTl);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = initialLBPTemplate(LBPTl);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }

    ImageBasicOp::copyToHost(inImage);

    // 提取子图像
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inImage, &insubimgCud);
    if (errcode != NO_ERROR){
        TemplateBasicOp::deleteTemplate(LBPTl);
        return errcode;
    }


    int index;
    int indexofinimg;
    int cx, cy;
    int inner[8];
    int outer[16];

    for (int r = 0;r < insubimgCud.imgMeta.height; r++) {
        for (int c = 0;c < insubimgCud.imgMeta.width; c++) {

            index = r * insubimgCud.imgMeta.width + c;     

            indexofinimg = r * insubimgCud.pitchBytes + c;   // 图像的宽度和矩阵的宽度不一样

            // 舍弃边缘
            if (c <= 2 || c >= insubimgCud.imgMeta.width - 3 || r <= 2 || r >= insubimgCud.imgMeta.height - 3) {
                descriptorOfCorner[index].inner = 0;
                descriptorOfCorner[index].outer = 0;
                continue;
            }

            if (isCorner[index] == 1) {
                // 提取内层循环
                for(int i = 0; i < 8; i++) {
                    cx = c + LBPTl->tplData[2 * i];
                    cy = r + LBPTl->tplData[2 * i + 1];
                    if (insubimgCud.imgMeta.imgData[cy * insubimgCud.pitchBytes + cx] > insubimgCud.imgMeta.imgData[indexofinimg]) {
                        inner[i] = 1;
                    }
                    else {
                        inner[i] = 0;
                    }
                }
                // 提取外层循环
                for(int i = 9; i < 25; i++) {
                    cx = c + LBPTl->tplData[2 * i];
                    cy = r + LBPTl->tplData[2 * i + 1];
                    if (insubimgCud.imgMeta.imgData[cy * insubimgCud.pitchBytes + cx] > insubimgCud.imgMeta.imgData[indexofinimg]) {
                        outer[i - 9] = 1;
                    }
                    else {
                        outer[i - 9] = 0;
                    }
                }
                descriptorOfCorner[index].inner = calArrayMinOnCPU(inner, 8);
                descriptorOfCorner[index].outer = calArrayMinOnCPU(outer, 16);
            }

            else {
                descriptorOfCorner[index].inner = 0;
                descriptorOfCorner[index].outer = 0;
            }

        }
    }

    TemplateBasicOp::deleteTemplate(LBPTl);

    return errcode; 
}

//*************************************************************************************************************

//*************************************************************************************************************

int LBPDescriptor::LBPMatchOnCPU(Descriptor *descriptorOfCorner1, Descriptor *descriptorOfCorner2,
                                Coordinate *matchCorner,  int width1, int height1, int width2, int height2)
{
    if (descriptorOfCorner1 == NULL || descriptorOfCorner2 == NULL || matchCorner == NULL) 
        return NULL_POINTER;

    int index1;
    int index2;
    int innerdistance;
    int outerdistance;

    for (int r = 0;r < height1; r++) {
        for (int c = 0;c < width1; c++) {

            index1 = r * width1 + c;

            matchCorner[index1].x = 0;
            matchCorner[index1].y = 0;
            if (descriptorOfCorner1[index1].inner == 0) {
                //continue;
            }
            else {
                for (int i = 0; i < height2; i++) {
                    for (int j = 0; j < width2; j++) {
                        index2 = i * width2 + j;
                        innerdistance = descriptorOfCorner1[index1].inner - descriptorOfCorner2[index2].inner;
                        innerdistance = (innerdistance > 0) ? innerdistance : -innerdistance;
                        outerdistance = descriptorOfCorner1[index1].outer - descriptorOfCorner2[index2].outer;
                        outerdistance = (outerdistance > 0) ? outerdistance : -outerdistance;
                        if (innerdistance < 1 && outerdistance < 1) {
                            matchCorner[index1].x = j;
                            matchCorner[index1].y = i;
                        }           
                    }
                }
            }

        }
    }

    return 0;
}

//*************************************************************************************************************

//*************************************************************************************************************
int LBPDescriptor::ImageMatchOfLBPOnCPU(Image *inImage1, Image *inImage2, int *isCorner1, int *isCorner2,
                                        Coordinate *matchCorner)
{
    Descriptor *descriptorOfCorner1 = new Descriptor[inImage1->width * inImage1->height];
    Descriptor *descriptorOfCorner2 = new Descriptor[inImage2->width * inImage2->height];



    calLBPDescriptorOnCPU(inImage1, isCorner1, descriptorOfCorner1);
    calLBPDescriptorOnCPU(inImage2, isCorner2, descriptorOfCorner2);



    LBPMatchOnCPU(descriptorOfCorner1, descriptorOfCorner2,matchCorner, 
                  inImage1->width, inImage1->height, inImage2->width, inImage2->height);

    return 0;

}