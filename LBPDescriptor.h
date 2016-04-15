// LBPDescriptor.h
// 创建者：丁燎原
// 
// LBP描述符（LBPDescriptor）
// 1、选择特征点的如下图邻域
// |      11,12,13     |
// |   10,         14  |
// |25,   1, 2, 3,   15|
// |24,   8, 9, 4,   16|
// |23,   7, 6, 5,   17|
// |   22,         18  |
// |      21,20,19     |
// 如果该点像素值大于特征点像素值，则将该点标1，否则标0。
// 2、然后将1-8号，10-25号组成的二进制串分别转换为十进制数，则这两个数为该点的特征点描述符
//    选择两圈的目的是增大区分度
// 3、为了克服旋转，可将1-8号和10-25号能够组成的最小二进制串分别转换为十进制数。例如：
//    0,0,0,1,0,0,0,0  可组成的最大二进制串为0,0,0,0,0,0,0,1。其余情况以此类推。
//    
// LBP具有平移和光照不变性，对于旋转具有一定的不变性。对于尺度、投影、仿射比较敏感。
// 由于LBP本身不具备作为特征点描述符的性质，且加上其维度太低，匹配效果不尽如人意。
// 实验表明对于平移，光照，旋转90度、180度、270度匹配效果较好。
// 
// 修订历史：
// 2014年12月30日（丁燎原）
// 		初始版本
// 2015年4月7日 （丁燎原）
//      添加注释
// 
#ifndef __LBPDESCRIPTOR_H__ 
#define __LBPDESCRIPTOR_H__

#include "Image.h"
#include "Template.h"
#include "DrawCircle.h"

// LBP特征点描述符
typedef struct Descriptor_st
{
	int inner;   // 内圈（1-8组成的最小十进制数）描述符
	int outer;   // 外圈（10-25组成的最小十进制数）描述符
}Descriptor;

class LBPDescriptor
{
private:

public:
	LBPDescriptor()
	{}
	// 计算图像每一个特征点的LBP描述符
	int calLBPDescriptor(
			Image *inImage,                  // 输入图像
			int *isCorner,                   // 判断矩阵
			Descriptor *descriptorOfCorner   // 描述符矩阵，传入前需要开辟空间
	);

	int calLBPDescriptorOnCPU(
			Image *inImage,                  // 输入图像
			int *isCorner,                   // 判断矩阵
			Descriptor *descriptorOfCorner   // 描述符矩阵，传入前需要开辟空间
	);

	int LBPMatch(
			Descriptor *descriptorOfCorner1, // 输入图像1的描述符矩阵
			Descriptor *descriptorOfCorner2, // 输入图像2的描述符矩阵
		    Coordinate *matchCorner,         // 匹配矩阵（如果（2,5）位置存放（3,8），则图像1（2,5）与
		                                     // 图像2（3,8）匹配），传入前需要开辟空间

		    int width1,                      // 输入图像1的宽度
		    int height1,                     // 输入图像1的高度
		    int width2,                      // 输入图像2的宽度
		    int height2                      // 输入图像2的高度
	);

	int LBPMatchOnCPU(
			Descriptor *descriptorOfCorner1, // 输入图像1的描述符矩阵
			Descriptor *descriptorOfCorner2, // 输入图像2的描述符矩阵
		    Coordinate *matchCorner,         // 匹配矩阵（如果（2,5）位置存放（3,8），则图像1（2,5）与
		                                     // 图像2（3,8）匹配），传入前需要开辟空间

		    int width1,                      // 输入图像1的宽度
		    int height1,                     // 输入图像1的高度
		    int width2,                      // 输入图像2的宽度
		    int height2                      // 输入图像2的高度
	);

	int ImageMatchOfLBP(
			Image *inImage1,            // 输入图像1
			Image *inImage2,            // 输入图像2
			int *isCorner1,             // 输入图像1的判断矩阵
			int *isCorner2,             // 输入图像2的判断矩阵
			Coordinate *matchCorner     // 匹配矩阵（如果（2,5）位置存放（3,8），则图像1（2,5）与
		                                // 图像2（3,8）匹配），传入前需要开辟空间
	);

	int ImageMatchOfLBPOnCPU(
			Image *inImage1,            // 输入图像1
			Image *inImage2,            // 输入图像2
			int *isCorner1,             // 输入图像1的判断矩阵
			int *isCorner2,             // 输入图像2的判断矩阵
			Coordinate *matchCorner     // 匹配矩阵（如果（2,5）位置存放（3,8），则图像1（2,5）与
		                                // 图像2（3,8）匹配），传入前需要开辟空间
	);
	
};
#endif

