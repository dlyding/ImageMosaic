#include "DrawCircle.h"
#include "Harris.h"
#include "SomeFunction.h"
#include "LBPDescriptor.h"
#include "RansacForImage.h"
#include <iostream>
using namespace std;

int main()
{
    Image *inImage1, *inImage2, *outImage1, *outImage2, *outImage;
    ImageBasicOp::newImage(&inImage1);
    ImageBasicOp::newImage(&inImage2);
    ImageBasicOp::newImage(&outImage1);
    ImageBasicOp::newImage(&outImage2);
    ImageBasicOp::newImage(&outImage);

    ImageBasicOp::readFromFile("test1.bmp",inImage1);
    ImageBasicOp::readFromFile("test2.bmp",inImage2);
    //DrawCircle dc(100,255,300,300);
    //dc.drawCircle(inImage);
    //ImageBasicOp::writeToFile("out.bmp",inImage);
    //ImageBasicOp::deleteImage(inImage);
    
    cudaEvent_t start, stop;
    float Time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    Harris h;
    int size1 = inImage1->width * inImage1->height;
    int size2 = inImage2->width * inImage2->height;
    /*TwoDMatrix *cm1, *cm2;
    cm1 = new TwoDMatrix[size1];
    cm2 = new TwoDMatrix[size2];
    h.calChangeMatrix(inImage1, cm1);
    h.calChangeMatrix(inImage2, cm2);*/
    /*for (int i = 0; i < inImage1->height; i++) {
        for (int j = 0; j < inImage1->width; j++) {
            cout<<cm1[i * inImage1->width + j].x3<<" ";
        }
        cout<<endl;
    }*/

   /* float *roi1, *roi2;
    roi1 = new float[size1];
    roi2 = new float[size2];
    h.gaussianChangeMatrix(cm1, roi1, inImage1->width, inImage1->height);
    h.gaussianChangeMatrix(cm2, roi2, inImage2->width, inImage2->height);*/
    /*for (int i = 0; i < inImage1->height; i++) {
        for (int j = 0; j < inImage1->width; j++) {
            cout<<roi1[i * inImage1->width + j]<<" ";
        }
        cout<<endl;
    }*/
    
    cudaEventRecord(start, 0);
    int *ic1, *ic2;
    ic1 = new int[size1];
    ic2 = new int[size2];
    /*h.cornerDetection(roi1, ic1, inImage1->width, inImage1->height);
    h.cornerDetection(roi2, ic2, inImage2->width, inImage2->height);*/
    
    h.calHarris(inImage1, ic1);
    h.calHarris(inImage2, ic2);
    /*for (int i = 0; i < inImage1->height; i++) {
        for (int j = 0; j < inImage1->width; j++) {
            cout<<ic1[i * inImage1->width + j]<<" ";
        }
        cout<<endl;
    }
*/
    /*DrawCircle dc;
    for (int i = 0; i < inImage1->height; i++) {
        for (int j = 0; j < inImage1->width; j++) {
            if(ic1[i * inImage1->width + j] == 1) {
                dc.setCoordinate(j, i);
                dc.drawCircle(inImage1);
            }
        }
    }*/
    

    LBPDescriptor ld;
    /*Descriptor *doc1, *doc2;
    doc1 = new Descriptor[size1];
    doc2 = new Descriptor[size2];
    ld.calLBPDescriptor(inImage1, ic1, doc1);
    ld.calLBPDescriptor(inImage2, ic2, doc2);*/

/*    for (int i = 0; i < inImage1->height; i++) {
        for (int j = 0; j < inImage1->width; j++) {
            cout<<doc1[i * inImage1->width + j].inner<<" ";
        }
        cout<<endl;
    }*/
//以上没有问题
    Coordinate *mc;
    mc = new Coordinate[size1];
    /*ld.LBPMatch(doc1, doc2, mc, inImage1->width, inImage1->height, inImage2->width, inImage2->height);
*/
    ld.ImageMatchOfLBP(inImage1, inImage2, ic1, ic2, mc);
   
    
    
    //cout<<"asd"<<endl;

  /*  for (int i = 0; i < inImage1->height; i++) {
        for (int j = 0; j < inImage1->width; j++) {
            cout<<mc[i * inImage1->width + j].x<<" ";
        }
        cout<<endl;
    }*/

    /*imageConnect(inImage1, inImage2, outImage);

    for (int i = 0; i < inImage1->height; i++) {
        for (int j = 0; j < inImage1->width; j++) {
            if(mc[i * inImage1->width + j].x != 0) {
                drawLine(outImage, j, i, mc[i * inImage1->width + j].x + inImage1->width, 
                         mc[i * inImage1->width + j].y, 255);
            }
        }
    }*/
    //cout<<ImageBasicOp::writeToFile("out.bmp",inImage1)<<endl;
    //ImageBasicOp::deleteImage(inImage);
    //

    vector<MatchPoint> mpt;
    float *tf = new float[9];
    pickupMatchPoint(mc, inImage1->width, inImage1->height, mpt);
    cout<<mpt.size()<<endl;

    //cout<<"asd"<<endl;
    Ransac(mpt, tf, 0.1, 10000);

    

    for(int i = 0; i < 9; i++) {
        cout<<tf[i]<<endl;
    }

    float distancex;
    float distancey;

    

    imageTransform(inImage1, outImage1, tf, 9, distancex, distancey);

    imageTranslation(inImage2, outImage2, distancex, distancey);

    imageFusion(outImage1, outImage2, outImage);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cout<<Time<<endl;

    //********************************************************************************
    
    //********************************************************************************
    
    /*imageConnect(inImage1, inImage2, outImage);
    drawLine(outImage, 5, 150, 150, 15, 200);*/

    ImageBasicOp::writeToFile("out.bmp", outImage);

    ImageBasicOp::deleteImage(inImage1);
    ImageBasicOp::deleteImage(inImage2);
    ImageBasicOp::deleteImage(outImage1);
    ImageBasicOp::deleteImage(outImage2);
    ImageBasicOp::deleteImage(outImage);

    /*delete []cm1;
    delete []cm2;
    delete []roi1;
    delete []roi2;*/
    delete []ic1;
    delete []ic2;
    //delete []doc1;
    //delete []doc2;
    delete []mc;
    delete []tf;


//************************************************************************************

//************************************************************************************

    /*Image *inImage, *outImage;
    ImageBasicOp::newImage(&inImage);
    ImageBasicOp::newImage(&outImage);

    ImageBasicOp::readFromFile("test.bmp",inImage);

    float transformMatrix[9] = {1, 0.5, 0, 0.5, 0.5, 0, 0, 0, 1};

    imageTransform(inImage, outImage, transformMatrix, 9);*/

   /* imageFusion(inImage1, inImage2, outImage);

    ImageBasicOp::writeToFile("out.bmp", outImage);
    ImageBasicOp::deleteImage(inImage1);
    ImageBasicOp::deleteImage(inImage2);
    ImageBasicOp::deleteImage(outImage);
*/
    /*float input[32] = {20,4,28,14,2,2,10,12,2,3,10,13,5,6,10,20,30,1,38,11,12,14,11,12,23,11,31,21,10,23,35,12};
    float matrix[9];
    vector<MatchPoint> matchpoint;

    pickupMatchPoint(input, 8, matchpoint);
    cout<<matchpoint.size()<<endl;

    //calTransformMatrix(input, matrix);
    Ransac(matchpoint, matrix, 0.1, 1000);

    for(int i = 0; i < 9; i++) {
        cout<<matrix[i]<<endl;
    } */

    return 0;
}