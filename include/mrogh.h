/*
An implementation of MROGH descriptor

For more information, refer to:

Bin Fan, Fuchao Wu and Zhanyi Hu, Aggregating Gradient Distributions into Intensity Orders: A Novel Local Image Descriptor,
<EM>CVPR 2011</EM>,pp.2377-2384.

Copyright (C) 2011 Bin Fan <bfan@nlpr.ia.ac.cn> 
All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
at your option any later version.
See the GNU General Public License for more details.

*/

#ifndef MROGH_H
#define MROGH_H

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

struct OxKey 
{
	float x;
	float y;
	float a;
	float b;
	float c;
	float trans[4];
	float square;
};

struct Pixel 
{
	int grid_pos_x;
	int grid_pos_y;
	float gray;
	float orient_dx;
	float orient_dy;
	bool operator < (const Pixel &m1) const
	{
		return gray < m1.gray;
	};
};

OxKey* ReadKeyFile(const char* filename, int& keynum);
void CalcuTrans(OxKey* pKeys,int n);
int* Extract_MROGH(const OxKey &key, cv::Mat im, int nDir,int nOrder,int nRegion);
int* Extract_OGH(const OxKey &key,cv::Mat imSrc,int nDir,int nOrder,double scale,int patch_width);
void Norm_desc(float *desc, double illuThresh, int dim);
float get_image_value(cv::Mat pImg, float x, float y);
Pixel* Normalize_Patch(const OxKey &key,cv::Mat in,float scale,int patch_width,int &nPixels);

#endif
