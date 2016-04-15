// RansacForImage.h
// 创建人：丁燎原
// 
// 随机抽样一致RANdom SAmple Consensus（RANSAC）
// 
// RANSAC的基本假设是：
//（1）数据由“局内点”和“局外点”组成
//（2）“局内点”可以用一个模型来描述，“局外点”是不能适应该模型的数据；
// 局外点产生的原因有：噪声的极值；错误的测量方法；对数据的错误假设。
// RANSAC也做了以下假设：给定一组（通常很小的）局内点，存在一个可以估计模型参数的过程；
// 而该模型能够解释或者适用于局内点。
// 
// RANSAC求解图像变换矩阵的步骤：
// 1、随机选取4对匹配点，使用该4对匹配点计算出变换矩阵
// 2、然后使用该变换矩阵来评估其余匹配点，得到该变换矩阵的准确率
// 3、根据预先设定的循环次数，重复进行步骤1和2，每次产生的模型要么因为局内点太少而被舍弃，
//    要么因为比现有的模型更好而被选用。
// 4、循环结束后，准确率最高的那个变换矩阵即为所求变换矩阵
// 
// 注意：
// RANSAC算法并不能保证所求模型正确，只能保证它在已求得的模型中最优。
// 另外，如果有两个或以上的模型适应该数据，RANSAC不能求出
// 
// 修订历史:
// 2015年3月15日
// 		初始版本
// 	2015年4月8日
// 		添加注释
// 		
#ifndef __RANSACFORIMAGE_H__
#define __RANSACFORIMAGE_H__

#include "DrawCircle.h"
#include <vector>
using namespace std;

// 定义匹配对结构体
typedef struct MatchPoint_st
{
    int x1;       // 点1 x坐标
    int y1;       // 点1 y坐标
    int x2;       // 点2 x坐标
    int y2;       // 点2 y坐标
} MatchPoint;

// 从匹配矩阵中提取匹配点存放到vector中
int pickupMatchPoint(
		Coordinate *matchCorner,          // 匹配矩阵
		int width,                        // 匹配矩阵的宽度
		int height,                       // 匹配矩阵的高度
		vector<MatchPoint> &matchpoint    // 匹配点集向量的引用
);

// 函数重载，从数组中提取匹配点存放到vector中
int pickupMatchPoint(
		float *input,                    // 输入数组 
		int sampleNum,                   // 数组大小
		vector<MatchPoint> &matchpoint   // 匹配点集向量的引用
);

// 计算变换矩阵
// inPoint大小为 4 * 4，transformMatrix大小为 9。
int calTransformMatrix(
		float *inPoint,                  // 输入4对匹配点数组
		float *transformMatrix           // 大小为9的变换矩阵，传入前需开辟空间
);

// 评估模型的准确率
float calAccuracyOfModel(
		vector<MatchPoint> matchpoint,  // 匹配点集向量
		float *transformMatrix,         // 变换矩阵
		float deviation                 // 误差阈值
);

// RANSAC算法
int Ransac(
		vector<MatchPoint> matchpoint,  // 匹配点集向量
		float *transformMatrix,         // 输出变换矩阵，传入前需要开辟空间
		float deviation,                // 误差阈值
		int count                       // 采样次数
);

#endif