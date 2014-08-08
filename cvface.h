#ifndef _JUN_CVFACE_H_
#define _JUN_CVFACE_H_

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <string>
#include <vector>

//#define LOOCV_TEST
//#define DO_EQUALIZE

namespace cvface{

	class Trainer
	{
	public:
		void reset(){names.clear();images.clear();labels.clear();}
		void enterClass(const char* class_name);
		void addSample(const char* fname);

		void train(const char* saveto,int type);

		cv::Ptr<cv::FaceRecognizer> load(const char* fname,int type);
		std::string label(int i){if(i>=0&&i<names.size())return names[i];return "Unknown";}
		public:
		std::vector<std::string> names;

		std::vector<cv::Mat> images;
		std::vector<int> labels;
	};

	extern Trainer g_trainer;

	bool normalizeSample(const char* fname,const char* dst_fname);
	void saveAvatar(const char* fname,const char* dst_fname);
	void doCapture(const char* trained_model_fname,int model_type);
	bool loocvTest(const char* trained_model_fname,int model_type,const char* fname,int label);

}

#endif