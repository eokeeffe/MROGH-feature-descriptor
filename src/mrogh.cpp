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

#include <mrogh.h>
#include <stdio.h>
#include <functional>
#include <algorithm>


OxKey* ReadKeyFile(const char* filename, int& keynum)
{
	FILE* f = fopen(filename,"rt");
	if ( f == NULL )
	{
		printf("file does not exist, %s\n",filename);
		return NULL;
	}
	float temp;
	fscanf(f,"%f\n",&temp);
	int n_keys;
	fscanf(f,"%d\n",&n_keys);
	keynum = n_keys;
	OxKey* pKeys = new OxKey[n_keys];
	int i;
	for (i = 0;i < n_keys;i++)
	{
		fscanf(f,"%f %f %f %f %f",&pKeys[i].x,&pKeys[i].y,&pKeys[i].a,&pKeys[i].b,&pKeys[i].c);
		if (temp > 1.0)
		{
			int drop = 0;
			for(int j = 0;j < temp;j++)
				fscanf(f," %d",&drop);
		}
		fscanf(f,"\n");
	}
	fclose(f);
	return pKeys;
}

void CalcuTrans(OxKey* pKeys,int n)
{
	cv::Mat A = cv::Mat(2,2,CV_32FC1);
	cv::Mat EigenVals = cv::Mat(2,1,CV_32FC1);
	cv::Mat EigenVects = cv::Mat(2,2,CV_32FC1);
	cv::Mat EigenVals_sqrt_inv = cv::Mat(2,2,CV_32FC1);
	
	for (int i = 0;i < n;i++)
	{
		A.at<float>(0,0) = pKeys[i].a;
		A.at<float>(0,1) = pKeys[i].b;
		A.at<float>(1,0) = pKeys[i].b;
		A.at<float>(1,1) = pKeys[i].c;
		// A = U' * D * U;
		cv::eigen( A, EigenVals, EigenVects);
		// D = D^-0.5
		EigenVals_sqrt_inv.at<float>(0,0) = 1.0f / (float)sqrt(EigenVals.at<float>(0,0));
		EigenVals_sqrt_inv.at<float>(0,1) = 0;
		EigenVals_sqrt_inv.at<float>(1,0) = 0;
		EigenVals_sqrt_inv.at<float>(1,1) = 1.0f / (float)sqrt(EigenVals.at<float>(1,0));
		pKeys[i].square = EigenVals_sqrt_inv.at<float>(0,0) * EigenVals_sqrt_inv.at<float>(1,1);
		// U = D * U
		int esv_w = EigenVals_sqrt_inv.rows;
		int ev_h = EigenVects.cols;
		EigenVals_sqrt_inv *= EigenVects;
		cv::Mat EigenVects_T = EigenVects.t();
		// A = U' * (D * U)
		//cvMatMul(EigenVects,EigenVals_sqrt_inv,A);
		A = EigenVects_T * EigenVals_sqrt_inv;
		
		pKeys[i].trans[0] = A.at<float>(0,0);
		pKeys[i].trans[1] = A.at<float>(0,1);
		pKeys[i].trans[2] = A.at<float>(1,0);
		pKeys[i].trans[3] = A.at<float>(1,1);
	}
	A = NULL;
	EigenVals = NULL;
	EigenVects = NULL;
	EigenVals_sqrt_inv = NULL;
}

int* Extract_MROGH(const OxKey &key, cv::Mat im, int nDir,int nOrder,int nRegion)
{
	int i;
	int *desc = new int[nDir*nOrder*nRegion];
	for (i = 0;i < nRegion;i++)
	{
		int *tmp_desc = 0;
 		if( tmp_desc = Extract_OGH(key,im,nDir,nOrder,1.5*i+3,41) )
		{
			for (int j = 0;j < nDir*nOrder;j++)
			{
				desc[i*nDir*nOrder+j] = tmp_desc[j];
			}
			delete [] tmp_desc;
		}
		else
		{
			delete [] desc;
			return NULL;
		}
	}
	return desc;
}

int* Extract_OGH(const OxKey &key, cv::Mat imSrc,int nDir,int nOrder,double scale,int patch_width)
{
	int nPixels = 0;
	Pixel *pPixel_Array = Normalize_Patch(key,imSrc,scale,patch_width,nPixels);
	if( pPixel_Array == NULL ) return NULL;
	std::sort(pPixel_Array,pPixel_Array+nPixels);

	float *desc = new float[nOrder*nDir];
	memset(desc,0,sizeof(float)*nOrder*nDir);
	int i,j;
	for (i = 0;i < nOrder;i++)
	{
		int gap = int(nPixels / double(nOrder) + 0.5);
		for (j = 0;j < nPixels;j++)
		{
			int idx_thresh_low = gap*i;
			int idx_thresh_high = gap*(i+1);
			if (idx_thresh_high > nPixels-1){
			    idx_thresh_high = nPixels-1;
			}
			if (pPixel_Array[j].gray < pPixel_Array[idx_thresh_low].gray){
			    continue;
			}
			if (pPixel_Array[j].gray > pPixel_Array[idx_thresh_high].gray){
			    break;
			}
			double dir = atan2(pPixel_Array[j].orient_dy,pPixel_Array[j].orient_dx);
			double mag = pPixel_Array[j].orient_dy * pPixel_Array[j].orient_dy
				+ pPixel_Array[j].orient_dx * pPixel_Array[j].orient_dx;
			mag = sqrt(mag);
			double idxDir = (dir + CV_PI) * nDir / (2.0 * CV_PI);
			if ((int)idxDir == nDir){
			    idxDir -= nDir;
			}
			int dirIdx[2];
			float dirWeight[2];
			dirIdx[0] = (int)idxDir;
			dirIdx[1] = (dirIdx[0] + 1) % nDir;
			dirWeight[0] = 1.0 - (idxDir - dirIdx[0]);
			dirWeight[1] = idxDir - dirIdx[0];
	
			desc[i*nDir+dirIdx[0]] += dirWeight[0] * mag;
			desc[i*nDir+dirIdx[1]] += dirWeight[1] * mag;
		}
	}
	Norm_desc(desc,0.2,nOrder*nDir);

	delete [] pPixel_Array;

	int *desc1 = new int[nOrder*nDir];
	for (i = 0;i < nOrder*nDir;i++)
	{
		desc1[i] = (int)(desc[i] * 255 + 0.5);
	}
	delete [] desc;

	return desc1;
}

void Norm_desc(float *desc, double illuThresh, int dim)
{
	// Normalize the descriptor, and threshold 
	// value of each element to 'illuThresh'.
	int i;
	double norm = 0.0;
	
	for (i=0; i<dim; ++i)
	{
		norm += desc[i] * desc[i];
	}
	
	norm = sqrt(norm);
	
	for (i=0; i<dim; ++i)
	{
		desc[i] /= norm;
		
		if (desc[i] > illuThresh)
		{
			desc[i] = illuThresh;
		}
	}
	
	// Normalize again.
	
	norm = 0.0;
	
	for (i=0; i<dim; ++i)
	{
		norm += desc[i] * desc[i];
	}
	
	norm = sqrt(norm);
	
	for (i=0; i<dim; ++i)
	{
		desc[i] /= norm;
	}
}

Pixel* Normalize_Patch(const OxKey &key, cv::Mat in, float scale, int patch_width, int &nPixels)
{
	float trans[4];
	trans[0] = key.trans[0] * (2.0 * scale / patch_width);
	trans[1] = key.trans[1] * (2.0 * scale / patch_width);
	trans[2] = key.trans[2] * (2.0 * scale / patch_width);
	trans[3] = key.trans[3] * (2.0 * scale / patch_width);
	int minX = in.cols;
	int maxX = 0;
	int minY = in.rows;
	int maxY = 0;
	double theta_interval = 5 * CV_PI / 180;
	for (int i = 0;i < 72;i++)
	{
		double xS = (1.414 * (patch_width / 2.0) + 8) * cos(theta_interval * i);
		double yS = (1.414 * (patch_width / 2.0) + 8) * sin(theta_interval * i);
		double x_trans = trans[0] * xS + trans[1] * yS + key.x;
		double y_trans = trans[2] * xS + trans[3] * yS + key.y;
		if (int(x_trans) < minX) minX = int(x_trans);
		if ((int(x_trans)+1) > maxX) maxX = int(x_trans) + 1;
		if (int(y_trans) < minY) minY = int(y_trans);
		if ((int(y_trans)+1) > maxY) maxY = int(y_trans) + 1;
	}
	minX = minX < 0 ? 0 : minX;
	minY = minY < 0 ? 0 : minY;
	maxX = maxX > (in.cols - 1) ? (in.cols - 1) : maxX;
	maxY = maxY > (in.rows - 1) ? (in.rows - 1) : maxY;
	int regionW = maxX - minX + 1;
	int regionH = maxY - minY + 1;
	
	cv::Rect rc(minX,minY,regionW,regionH);
	//cvSetImageROI(in,rc);
	
	cv::Mat image_roi(in,rc);
	cv::Mat in_smooth = cv::Mat(regionW,regionH,CV_8U);
	
 	if ( key.square * scale * scale > (patch_width * patch_width / 4.0) )
 	{
 		double sigma = key.square * scale * scale / ((patch_width * patch_width / 4.0));
 		sigma = sqrt(sigma);
 		//cvSmooth(in,in_smooth,CV_GAUSSIAN,5,5,sigma);
 		cv::GaussianBlur( image_roi, in_smooth, cv::Size( 5, 5 ), sigma, 0 );
 	}
	else
	{
	    in_smooth = image_roi.clone();
	}
    
	int patch_radius = patch_width / 2;
	int x,y;
	
	cv::Mat outPatch = cv::Mat(cv::Size(patch_radius*2+1+16,patch_radius*2+1+16),CV_32FC1);
	
	float *out_data = (float*)outPatch.data;
	for (y = -patch_radius-8;y <= patch_radius+8;y++)
	{
		for (x = -patch_radius-8;x <= patch_radius+8;x++)
		{
			float x1 = trans[0] * x + trans[1] * y + key.x;
			float y1 = trans[2] * x + trans[3] * y + key.y;
			x1 -= minX;
			y1 -= minY;
			if (x1 < 0 || x1 > (in_smooth.cols - 1) || y1 < 0 || y1 > (in_smooth.rows - 1))
			{
				out_data[(y + patch_radius + 8) * outPatch.cols + (x + patch_radius + 8)] = 0;
			}
			else
			{
				out_data[(y + patch_radius + 8) * outPatch.cols + (x + patch_radius + 8)] = 
					get_image_value(in_smooth,x1,y1);
			}
		}
	}
	
	cv::GaussianBlur( outPatch, outPatch, cv::Size( 5, 5 ), 1.6, 0 );
	
	Pixel *pPixel_Array = new Pixel[patch_width * patch_width - 1];
	int nCount = 0;
	for (y = -patch_radius;y <= patch_radius;y++)
	{
		for (x = -patch_radius;x <= patch_radius;x++)
		{
			double orient_dx, orient_dy;
			if( 0 == y && 0 == x) continue;
			double dis = x * x + y * y;
			dis = sqrt(dis);
			if (dis > patch_radius) continue;
			float x_ori = trans[0] * x + trans[1] * y + key.x;
			float y_ori = trans[2] * x + trans[3] * y + key.y;
			x_ori -= minX;
			y_ori -= minY;

			double theta = atan2((double)y,(double)x);
			float r = 4;
			
			float x1 = x + r * cos(theta);
			float y1 = y + r * sin(theta);
			float trans_x = trans[0] * x1 + trans[1] * y1 + key.x;
			float trans_y = trans[2] * x1 + trans[3] * y1 + key.y;
			trans_x -= minX;
			trans_y -= minY;
			if (trans_x < 0 || trans_x > (in_smooth.cols - 1) || 
			    trans_y < 0 || trans_y > (in_smooth.rows - 1)){
			    continue;
			 }
			orient_dx = get_image_value(in_smooth,trans_x,trans_y);

			x1 = x - r * cos(theta);
			y1 = y - r * sin(theta);
			trans_x = trans[0] * x1 + trans[1] * y1 + key.x;
			trans_y = trans[2] * x1 + trans[3] * y1 + key.y; 
			trans_x -= minX;
			trans_y -= minY;
			if (trans_x < 0 || trans_x > (in_smooth.cols - 1)  || 
			    trans_y < 0 || trans_y > (in_smooth.rows - 1)){
			    continue;
			}
			orient_dx -= get_image_value(in_smooth,trans_x,trans_y);
		
			x1 = x - r * sin(theta);
			y1 = y + r * cos(theta);
			trans_x = trans[0] * x1 + trans[1] * y1 + key.x;
			trans_y = trans[2] * x1 + trans[3] * y1 + key.y; 
			trans_x -= minX;
			trans_y -= minY;
			if (trans_x < 0 || trans_x > (in_smooth.cols - 1) || 
			    trans_y < 0 || trans_y > (in_smooth.rows - 1)){
			    continue;
			}
			orient_dy = get_image_value(in_smooth,trans_x,trans_y);
			
			x1 = x + r * sin(theta);
			y1 = y - r * cos(theta);
			trans_x = trans[0] * x1 + trans[1] * y1 + key.x;
			trans_y = trans[2] * x1 + trans[3] * y1 + key.y; 
			trans_x -= minX;
			trans_y -= minY;
			if (trans_x < 0 || trans_x > (in_smooth.cols - 1) || 
			    trans_y < 0 || trans_y > (in_smooth.rows - 1)){
			    continue;
			}
			orient_dy -= get_image_value(in_smooth,trans_x,trans_y);
			
			pPixel_Array[nCount].orient_dx = orient_dx;
			pPixel_Array[nCount].orient_dy = orient_dy;			
			pPixel_Array[nCount].grid_pos_x = x;
			pPixel_Array[nCount].grid_pos_y = y;
			pPixel_Array[nCount].gray = out_data[(y + patch_radius + 8) * outPatch.cols + (x + patch_radius + 8)];
		
			nCount++;
		}
	}
	nPixels = nCount;
	
	outPatch = NULL;
	in_smooth = NULL;
	return pPixel_Array;
}

float get_image_value(cv::Mat pImg, float x, float y)
{	
	int x1 = (int)x;
	int y1 = (int)y;
	int x2 = x1 + 1;
	int y2 = y1 + 1;
	float gray = 0;

	if( (x2 - x) * (y2 - y) != 0 ){
	    gray += (x2 - x) * (y2 - y) * pImg.at<uchar>(y1,x1)/255.0f;
	}
	if( (x - x1) * (y2 - y) != 0 ){
	    gray += (x - x1) * (y2 - y) * pImg.at<uchar>(y1,x2)/255.0f;
	}
	if( (x2 - x) * (y - y1) != 0 ){
	    gray += (x2 - x) * (y - y1) * pImg.at<uchar>(y2,x1)/255.0f;
	}
	if( (x - x1) * (y - y1) != 0 ){
	    gray += (x - x1) * (y - y1) * pImg.at<uchar>(y2,x2)/255.0f;
	}
	
	return gray;
}
