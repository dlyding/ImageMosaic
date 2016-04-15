#include "RansacForImage.h"
#include "Image.h"
#include "CovarianceMatrix.h"
#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace Eigen;

// 计算一点变换后的坐标的宏定义
#define calz(matrix, x, y) ((x) * matrix[6] + (y) * matrix[7] + matrix[8])
#define calx(matrix, x, y) ((((x) * matrix[0] + (y) * matrix[1] + matrix[2])) / (calz(matrix, x, y)))
#define caly(matrix, x, y) ((((x) * matrix[3] + (y) * matrix[4] + matrix[5])) / (calz(matrix, x, y)))
// 绝对值计算宏定义
#define abs(x) ((x) > 0 ? (x) : -1 * (x))

int pickupMatchPoint(Coordinate *matchCorner, int width, int height, vector<MatchPoint> &matchpoint)
{
	MatchPoint mp;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			// 判断是否有匹配点
			if (matchCorner[i * width + j].x != 0 || matchCorner[i * width + j].y != 0) {
				mp.x1 = j;
				mp.y1 = i;
				mp.x2 = matchCorner[i * width + j].x;
				mp.y2 = matchCorner[i * width + j].y;
				matchpoint.push_back(mp);
			}
		}
	}
	return 0;
}

int pickupMatchPoint(float *input, int sampleNum, vector<MatchPoint> &matchpoint)
{
	MatchPoint mp;
	for (int i = 0; i < sampleNum; i++) {
		mp.x1 = input[i * 4 + 0];
		mp.y1 = input[i * 4 + 1];
		mp.x2 = input[i * 4 + 2];
		mp.y2 = input[i * 4 + 3];
		matchpoint.push_back(mp);
	}
	return 0;
}

// inPoint大小为 4 * 4，transformMatrix大小为 9。
int calTransformMatrix(float *inPoint, float *transformMatrix)   
{
	float middleMatrix[8 * 8];
	//float *inverseMatrix = new float[8 * 8];
	MatrixXf middleM(8, 8);
	int middleArray[8];
	for(int i = 0; i < 8; i++) {
		if(i % 2 == 0){
			middleMatrix[i * 8] = inPoint[i / 2 * 4];
			middleMatrix[i * 8 + 1] = inPoint[i / 2 * 4 + 1];
			middleMatrix[i * 8 + 2] = 1;
			middleMatrix[i * 8 + 3] = 0;
			middleMatrix[i * 8 + 4] = 0;
			middleMatrix[i * 8 + 5] = 0;
			middleMatrix[i * 8 + 6] = -1 * inPoint[i / 2 * 4] * inPoint[i / 2 * 4 + 2];
			middleMatrix[i * 8 + 7] = -1 * inPoint[i / 2 * 4 + 1] * inPoint[i / 2 * 4 + 2];
			middleArray[i] = inPoint[i / 2 * 4+ 2];
		}
		else {
			middleMatrix[i * 8] = 0;
			middleMatrix[i * 8 + 1] = 0;
			middleMatrix[i * 8 + 2] = 0;
			middleMatrix[i * 8 + 3] = inPoint[i / 2 * 4];
			middleMatrix[i * 8 + 4] = inPoint[i / 2 * 4 + 1];
			middleMatrix[i * 8 + 5] = 1;
			middleMatrix[i * 8 + 6] = -1 * inPoint[i / 2 * 4] * inPoint[i / 2 * 4 + 3];
			middleMatrix[i * 8 + 7] = -1 * inPoint[i / 2 * 4 + 1] * inPoint[i / 2 * 4 + 3];
			middleArray[i] = inPoint[i / 2 * 4+ 3];
		}
		transformMatrix[i] = 0;
	}

	for(int i = 0; i < 8; i++) {
		for(int j = 0; j < 8; j++) {
			middleM(i, j) = middleMatrix[i * 8 + j]; 
		}
	}

	/*for(int i = 0; i < 8; i++) {
		for(int j = 0; j < 8; j++) {
			cout<<middleMatrix[i * 8 + j]<<"  "; 
		}
		cout<<endl;
	}
	for(int i = 0; i < 8; i++) {
		for(int j = 0; j < 8; j++) {
			cout<<middleM(i, j)<<"  "; 
		}
		cout<<endl;
	}*/

	//calInverseMatrix(middleMatrix, inverseMatrix, 8, 8);
	middleM = middleM.inverse();

	/*for(int i = 0; i < 8; i++) {
		for(int j = 0; j < 8; j++) {
			cout<<middleM(i, j)<<"  "; 
		}
		cout<<endl;
	}*/

/*	 for(int i = 0; i < 8; i++) {
        cout<<middleArray[i]<<endl;
    } */

	for(int i = 0; i < 8; i++) {
		for(int j = 0; j < 8; j++) {
			transformMatrix[i] += (middleM(i, j) * middleArray[j]); 
		}
	}
	transformMatrix[8] = 1;

	//delete []inverseMatrix;
	return 0;
}

float calAccuracyOfModel(vector<MatchPoint> matchpoint, float *transformMatrix, float deviation)
{
	int correctPoint = 0;
	float x;
	float y;
	for(int i = 0; i < matchpoint.size(); i++) {
		x = calx(transformMatrix, matchpoint.at(i).x1, matchpoint.at(i).y1);
		y = caly(transformMatrix, matchpoint.at(i).x1, matchpoint.at(i).y1);
		if(abs(x - matchpoint.at(i).x2) < deviation && abs(y - matchpoint.at(i).y2) < deviation) {
			correctPoint++;
		}
	}
	return (correctPoint * 1.0 / matchpoint.size());
}

int Ransac(vector<MatchPoint> matchpoint, float *transformMatrix, float deviation, int count)
{
	float accuracy = 0;
	float accuracyTemp;
	float *transformMatrixTemp = new float[9];
	float *inPoint = new float[4 * 4];
	int random;

	// 注意随机数种子生成函数的摆放位置
	srand((int)time(NULL));
	
	for (int i = 0; i < count; i++) {
		for (int j = 0; j < 4; j++) {

			random = rand() % matchpoint.size();

			//cout<<random<<" ";

			inPoint[j * 4 + 0] = matchpoint.at(random).x1;
			inPoint[j * 4 + 1] = matchpoint.at(random).y1;
			inPoint[j * 4 + 2] = matchpoint.at(random).x2;
			inPoint[j * 4 + 3] = matchpoint.at(random).y2;
		}
		//cout<<endl;
		calTransformMatrix(inPoint, transformMatrixTemp);
		accuracyTemp = calAccuracyOfModel(matchpoint, transformMatrixTemp, deviation);

/*		for(int i = 0; i < 9; i++) {
        	cout<<transformMatrixTemp[i]<<endl;
    	} */

		if(accuracy < accuracyTemp) {
			accuracy = accuracyTemp;
			memcpy(transformMatrix, transformMatrixTemp, 9 * sizeof(float));
		}
	}

	delete []transformMatrixTemp;
	delete []inPoint;

	return 0;
}