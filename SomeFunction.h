// SomeFunction.h
// 创建者：丁燎原
// 
// 一些辅助函数（SomeFunction）
// 
// 修订历史：
// 2014年12月20日（丁燎原）
// 		初始版本
// 2015年4月7日 （丁燎原）
// 		添加注释，修改图像连接一处bug
// 
#ifndef __SOMEFUNCTION_H__
#define __SOMEFUNCTION_H__

#include "Image.h"

// 图像连接
int imageConnect(
		Image *inImg1,  // 输入图像一
		Image *inImg2,  // 输入图像二
		Image *outImg   // 输出图像
); 

// 画直线
int drawLine(
		Image *inImg,   // 输入图像
		int x1,         // 起点x坐标
		int y1,         // 起点y坐标
		int x2,         // 终点x坐标
		int y2,         // 终点y坐标
		int pixel       // 直线像素值
);

// 图像变换
// 不包含投影变换
int imageTransform(
		Image *inImg,             // 输入图像
		Image *outImg,            // 输出变换后的图像
		float *transformMatrix,   // 变换矩阵
		int matrixSize,           // 变换矩阵的大小
		float &distancex,         // 水平平移距离
		float &distancey          // 竖直平移距离
); 

//平移变换，主要为了调整图片
int imageTranslation(
		Image *inImg,              // 输入图像
		Image *outImg,             // 输出变换后的图像
		float distancex,           // 水平平移距离
		float distancey            // 竖直平移距离
); 

// 图像融合
int imageFusion(
		Image *inImg1,             // 输入图像1 
		Image *inImg2,             // 输入图像2
		Image *outImg              // 输出图像
);

#endif