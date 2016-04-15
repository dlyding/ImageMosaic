#include "Image.h"
#include "Template.h"
#include "ErrorCode.h"
#include "DrawCircle.h"
#include <iostream>
#include <vector>
using namespace std;

int DrawCircle::_CircleTemplate(Template * tp)
{
    if(tp == NULL)
        return NULL_POINTER;
    int errcode;
    vector<Coordinate> vt;
    Coordinate cdt;
    int x = 0, y = radius;
    int radius2 = radius * radius;

    // 处理与坐标轴相交的特殊情况
    cdt.x = 0;
    cdt.y = radius;
    vt.push_back(cdt);
    cdt.y = -radius;
    vt.push_back(cdt);
    cdt.x = radius;
    cdt.y = 0;
    vt.push_back(cdt);
    cdt.x = -radius;
    vt.push_back(cdt);

    // 整个迭代过程采用经典的八分圆迭代法，即只迭代推导出圆的右上 1/8 部分（依
    // 右手坐标来说），即从 (0, raidus) 至 (sqrt(radius), sqrt(radius)) 段的
    // 点，其余的点都通过圆自身的对称性映射得到。如果迭代得到 (x, y) 为圆上一
    // 点，那么 (-x, y)、(x, -y)、(-x, -y)、(y, x)、(-y, x)、(y, -x)、(-y, -x)
    // 也都将是圆上的点。

    while(x < y) {

        // 计算下一个点。这里对于下一个点只有两种可能，一种是 (x + 1, y)，另一
        // 种是 (x + 1, y - 1)。具体选择这两种中的哪一个，要看它们谁更接近真实
        // 的圆周曲线。这段代码就是计算这两种情况距离圆周曲线的距离平方（开平方
        // 计算太过复杂，却不影响这里的结果，因此我们没有进行开平方计算，而使用
        // 距离的平方值作为判断的条件）。

        x++;
        int d1 = x * x + y * y - radius2;
        int d2 = x * x + (y - 1) * (y - 1) - radius2;
        d1 = (d1 > 0) ? d1 : -d1;
        d2 = (d2 > 0) ? d2 : -d2;

        if(d1 > d2) {
            y--;
        }
        if(x > y)
            break;
        cdt.x = x;
        cdt.y = y;
        vt.push_back(cdt);
        cdt.y = -y;
        vt.push_back(cdt);
        cdt.x = -x;
        vt.push_back(cdt);
        cdt.y = y;
        vt.push_back(cdt);
        if(x == y)
            break;
        cdt.x = y;
        cdt.y = x;
        vt.push_back(cdt);
        cdt.y = -x;
        vt.push_back(cdt);
        cdt.x = -y;
        vt.push_back(cdt);
        cdt.y = x;
        vt.push_back(cdt);
    }
    errcode = TemplateBasicOp::makeAtHost(tp, vt.size());
    if(errcode != NO_ERROR) {
        return errcode;
    }
    for(int i = 0; i < vt.size(); i++) {
        tp->tplData[2 * i] = vt.at(i).x;
        tp->tplData[2 * i + 1] = vt.at(i).y;  //指针++，释放时应注意
    }
    tp->count = vt.size();
    return NO_ERROR;
}

int DrawCircle::drawCircle(Image * inImage)
{
    if(inImage == NULL) 
        return NULL_POINTER;
    int errcode;
    Template *tp;
    errcode = TemplateBasicOp::newTemplate(&tp);
    if(errcode != NO_ERROR)
        return errcode;
    _CircleTemplate(tp);
    // 2015年4月22日（丁燎原）
    // 修改一处bug
    ImageBasicOp::copyToHost(inImage);
    ImageCuda insubimgCud;
    errcode = ImageBasicOp::roiSubImage(inImage, &insubimgCud);
    if(errcode != NO_ERROR) {
        TemplateBasicOp::deleteTemplate(tp);
        return errcode;
    }

    int dx,dy,index;
    for(int i = 0; i < tp->count; i++) {
        dx = targetCd.x + tp->tplData[2 * i];
        dy = targetCd.y + tp->tplData[2 * i + 1];
        if((dx < insubimgCud.imgMeta.width) && (dy < insubimgCud.imgMeta.height)) {
            index = dy * insubimgCud.pitchBytes + dx;
            insubimgCud.imgMeta.imgData[index] = pixel;
        }
    }
    TemplateBasicOp::deleteTemplate(tp);
    if (cudaGetLastError() != cudaSuccess)
        return CUDA_ERROR;
    return NO_ERROR;
}

