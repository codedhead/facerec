#include "cvface.h"

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <stdio.h>

#include <windows.h>

//#define FACE_DEBUG


using namespace std;
using namespace cv;

namespace cvface{

Trainer g_trainer;

String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
RNG rng(12345);

const int NORM_IMG_WIDTH=92;
const int NORM_IMG_HEIGHT=112;

const int AVATAR_IMG_WIDTH=46;
const int AVATAR_IMG_HEIGHT=56;

int initFaceCascade()
{
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); exit(-1); };
	return 1;
}

enum ENormalizeAction
{
	ENorm_RGB=0,
	ENorm_Gray=1,
	ENorm_Equalize=2
};

bool detectAndNormalize(Mat frame,Mat& res,int action=0)
{
	if(face_cascade.empty())
		initFaceCascade();

	res=frame;

	if( !frame.empty() )
	{
		std::vector<Rect> faces;
		Mat frame_gray;

		cvtColor( frame, frame_gray, CV_BGR2GRAY );
		//equalizeHist( frame_gray, frame_gray );

		face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
		if(faces.size()==0)
		{
			return false;
		}

		// first face
		Mat faceROI = frame( faces[0] ); // rgb
		cv::resize(faceROI,faceROI,cv::Size(NORM_IMG_WIDTH,NORM_IMG_HEIGHT));

		res=faceROI;

		if(action>=ENorm_Gray)
			cvtColor( res, res, CV_BGR2GRAY );
		if(action>ENorm_Equalize)
			equalizeHist( res, res );

		return true;
	}
	return false;
}

void saveAvatar(const char* fname,const char* dst_fname)
{
	Mat frame;
	double scale = 1.3;

	//--  Load image
	frame = imread( fname, 1 );

	Mat faceROI;
	if(!detectAndNormalize(frame,faceROI))
	{
		printf("No face detected in \"%s\"\n",fname);
	}
	else
	{
		if(!imwrite(dst_fname,faceROI))
		{
			printf("Fail to save file: \"%s\"\n",dst_fname);
		}
	}
}

bool normalizeSample(const char* fname,const char* dst_fname)
{
	Mat frame;
	double scale = 1.3;

	//--  Load image
	frame = imread( fname, 1 );

	Mat faceROI;
	if(!detectAndNormalize(frame,faceROI,ENorm_Gray))
	{
		printf("No face detected in \"%s\"\n",fname);
	}
	else
	{
		if(!imwrite(dst_fname,faceROI))
		{
			printf("Fail to save file: \"%s\"\n",dst_fname);
		}
		else
		{
			return true;
		}
	}
	return false;
}

void Trainer::enterClass(const char* class_name)
{
	names.push_back(string(class_name));

#ifndef LOOCV_TEST
	printf("\nadding new class: %s\n",class_name);
#endif
}
void Trainer::addSample(const char* fname)
{
	int class_count=names.size();
	if(class_count<=0)
	{
		printf("class_count=0, cannot load image before adding a class\n");
		return;
	}

	Mat img=imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
	//cvtColor( img, img, CV_BGR2GRAY );
	if(img.empty())
	{
		printf("Fail to read image: \"%s\"\n",fname);
	}
	else
	{
#ifdef DO_EQUALIZE
		equalizeHist( img, img );
#endif
		images.push_back(img);
		labels.push_back(class_count-1);
#ifndef LOOCV_TEST
		putchar('*');
#endif
	}
}

void Trainer::train(const char* saveto,int type)
{
	Ptr<FaceRecognizer> model;
	if(type==0)
		model=createFisherFaceRecognizer();
	else
		model=createLBPHFaceRecognizer();

	
	printf("\nbegin training...\n");
	model->train(images, labels);
	model->save(saveto);

	char lable_fname[MAX_PATH];
	sprintf(lable_fname,"%s.labels",saveto);
	FILE* fp=fopen(lable_fname,"wt");
	fprintf(fp,"%d\n",names.size());
	for(int i=0,iLen=names.size();i<iLen;++i)
	{
		fprintf(fp,"%s\n",names[i].c_str());
	}
	fclose(fp);
	printf("Trained model saved to \"%s\"\n",saveto);
}

Ptr<FaceRecognizer> Trainer::load(const char* fname,int type)
{
	Ptr<FaceRecognizer> model;
	if(type==0)
		model=createFisherFaceRecognizer();
	else
		model=createLBPHFaceRecognizer();

	model->load(fname);	


	char lable_fname[MAX_PATH];
	sprintf(lable_fname,"%s.labels",fname);
	FILE* fp=fopen(lable_fname,"rt");
	if(fp==0)
	{
		printf("Fail to load .labels file, please re-train the model\n");
		return NULL;
	}
	int class_count=0;
	fscanf(fp,"%d",&class_count);fgetc(fp);

	names.clear();
	char buf[256];
	for(int i=0;i<class_count;++i)
	{
		fscanf(fp,"%s",buf);
		names.push_back(buf);
	}
	fclose(fp);
	
	return model;
}

#define CAPTURE_WND_NAME "Capture"


void drawAonB(Mat& a,Mat& b,const Point& pt)
{
	if(a.empty()||b.empty()) return;

	int w=min(max(b.cols-1-pt.x,0),a.cols),
	h=min(max(b.rows-1-pt.y,0),a.rows);
	cv::Rect roi( pt, cv::Size(w,h));
	a(Rect(Point(0,0),Size(w,h))).copyTo( b(roi) );
}


void detectAndRecognize( Mat frame, Ptr<FaceRecognizer> model )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	//equalizeHist( frame_gray, frame_gray );

	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	//faces.push_back(Rect(Point(0,0),Size(frame.cols,frame.rows)));
	for( size_t i = 0; i < faces.size(); i++ )
	{
		Point bottom_right(faces[i].x + faces[i].width,faces[i].y + faces[i].height);
		Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
		ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

		Mat to_rec = frame_gray( faces[i] );
		cv::resize(to_rec,to_rec,cv::Size(NORM_IMG_WIDTH,NORM_IMG_HEIGHT));
#ifdef DO_EQUALIZE
		equalizeHist(to_rec,to_rec);
#endif

#ifdef FACE_DEBUG
		Mat rgb_to_rec;
		cvtColor( to_rec, rgb_to_rec, CV_GRAY2BGR );
		drawAonB(rgb_to_rec,frame,bottom_right/*+Point(0,NORM_IMG_HEIGHT)*/);
#endif

		int predictLabel=-1;
		double confidence=0.;
		predictLabel=model->predict(to_rec);
		//printf("confidence: %lf\n",confidence);

		if(predictLabel==-1) continue;
		
 		string class_name=g_trainer.label(predictLabel);

		
				
		Mat avatar=imread(string("data/")+class_name+"/"+class_name+".avatar");
#ifndef FACE_DEBUG
		drawAonB(avatar,frame,bottom_right);
#endif
// 		if(!avatar.empty())
// 		{
// 			int w=min(max(frame.cols-1-bottom_right.x,0),avatar.cols),
// 				h=min(max(frame.rows-1-bottom_right.y,0),avatar.rows);
//  			cv::Rect avatar_roi( bottom_right, cv::Size(w,h));
//  			avatar(Rect(Point(0,0),Size(w,h))).copyTo( frame(avatar_roi) );
// 		}		
		
		putText(frame,class_name,bottom_right,FONT_HERSHEY_SIMPLEX, 1.5, cvScalar(250,20,10),3);
	}

	//-- Show what you got
	imshow( CAPTURE_WND_NAME, frame );
}

bool doing_capture=false;
int frame_cnt=0;
void doCapture(const char* trained_model_fname,int model_type)
{
	if(doing_capture) return;	

	if(face_cascade.empty())
		initFaceCascade();

	Ptr<FaceRecognizer> model=g_trainer.load(trained_model_fname,model_type);

	if(model.empty())
	{
		printf("Fail to load trained model, please re-train the model\n");
		return;
	}

	frame_cnt=0;
	doing_capture=true;
	printf("Start capturing, press Q to stop\n");

	CvCapture* capture=0;
	Mat frame;
	
	capture = cvCaptureFromCAM( 0 );
	if( capture )
	{
		while( true )
		{
			//double t = (double)cvGetTickCount();
			frame = cvQueryFrame( capture );
			
			if( !frame.empty() )
			{ 
				detectAndRecognize(frame,model);
			}
			else
			{ printf(" --(!) No captured frame -- Break!"); break; }

			//t = (double)cvGetTickCount() - t;printf( "time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

			int c = waitKey(10);
			if( (char)c == 'q' ) { break; }
			
			++frame_cnt;
		}
		cvReleaseCapture(&capture);
	}
	else
	{
		printf("Fail to access the Camera\n");
	}
	cvDestroyWindow(CAPTURE_WND_NAME);

	doing_capture=false;
}



bool loocvTest(const char* trained_model_fname,int model_type,const char* fname,int label)
{
	if(face_cascade.empty())
		initFaceCascade();

	Ptr<FaceRecognizer> model=g_trainer.load(trained_model_fname,model_type);

	if(model.empty())
	{
		printf("Fail to load trained model, please re-train the model\n");
		return false;
	}



	Mat frame=imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
	if(frame.empty())
	{
		printf("Fail to load loocv file\n");
		return false;
	}

	int predictLabel=-1;
	double confidence=0.;
	predictLabel=model->predict(frame);

	string class_name=g_trainer.label(predictLabel);
	printf("predict: %s\n",class_name.c_str());

	return (predictLabel==label);
}


}