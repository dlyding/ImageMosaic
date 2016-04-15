// DrawCircle.h
// 创建者：丁燎原
// 
// 画圆（drawCircle）
// 
// 修订历史：
// 2014年12月5日（丁燎原）
//      初始版本
// 2015年4月22日（丁燎原）
//      修改一处bug
// 
#ifndef __DRAWCIRCLE_H__
#define __DRAWCIRCLE_H__

#include "Image.h"
#include "Template.h"

// 定义坐标结构体
typedef struct Coordinate_st
{
    int x;       // x坐标
    int y;       // y坐标
} Coordinate;

//画圆类
class DrawCircle
{
private:
    int radius;           // 半径
    int pixel;            // 画圆像素值
    Coordinate targetCd;  // 圆心坐标

    int _CircleTemplate(Template * tp);   // 构造圆形模板
public:
    DrawCircle() {
        radius = 6;
        pixel = 255;
        targetCd.x = 0;
        targetCd.y = 0;
    }

    DrawCircle(int radius, int pixel, int x, int y) {
        this->radius = (radius < 1) ? 1 : radius;
        this->pixel = (pixel < 0 || pixel > 255) ? 255 : pixel;
        targetCd.x = (x < 0) ? 0 : x;
        targetCd.y = (y < 0) ? 0 : y;
    }

    int getRadius() {
        return radius;
    }

    int getPixel() {
        return pixel;
    }

    void setRadius(int radius) {
        this->radius = (radius < 1) ? 1 : radius;
    }

    void setPixel(int pixel) {
        this->pixel = (pixel < 0 || pixel > 255) ? 255 : pixel;
    }

    void setCoordinate(int x, int y) {
        targetCd.x = (x < 0) ? 0 : x;
        targetCd.y = (y < 0) ? 0 : y;
    }

    int drawCircle(Image * inImage);

    //int drawCircle(Image * inImage, Image * outImage);

    ~DrawCircle() {}
};

#endif