#include "stdafx.h"
#include "OmniStitcherR.h"
#include <thread>
#include <mutex>


const std::string CAMERA_MATRIX_IDEN = "cameraMatrix";
const std::string DIST_COEFF_IDEN = "distCoeffs";

OmniStitcherR::OmniStitcherR(void)
{
	hasCalibrationData_ = false;
	hasInputFolderData_ = false;
	hasOutputFolderData_ = false;
	hasCameraParams_ = false;
	isCameraUndistortMapComputed_ = false;
	workScale_ = 0.5;
}


OmniStitcherR::~OmniStitcherR(void)
{
}

void OmniStitcherR::StitchFrame(std::string filename)
{
	if(!IsReadyForStitching())
		return;

	std::vector<Mat> images(NUM_OF_CAMERAS);
	ReadFrame(filename, images);

}

void OmniStitcherR::EstimateCameraParameters(const std::vector<std::string> &testFiles)
{
	std::vector<detail::ImageFeatures> imageFeaturesAcc(NUM_OF_CAMERAS);
	std::vector<std::vector<DMatch>> matchesAcc(NUM_OF_CAMERAS);

	for (auto &frame : testFiles)
	{
		std::cout << "Extracting data from file " << frame << std::endl;
		std::vector<Mat> images(NUM_OF_CAMERAS);
		ReadFrame(frame, images);
		std::vector<detail::ImageFeatures> imageFeatures;
		std::vector<std::vector<DMatch>> matches;
		MatchFrame(images, imageFeatures, matches);
		AppendMatchingData(imageFeatures, matches, imageFeaturesAcc, matchesAcc);
	}

	rotationVec.resize(NUM_OF_CAMERAS);
	translationVec.resize(NUM_OF_CAMERAS);
	//TO-DO parallelize
	cv::Mat rVecAB, rVecBC, rVecCD, rVecDE, rVecEA;
	cv::Mat tVecAB, tVecBC, tVecCD, tVecDE, tVecEA;
	EstimateCameraRotation(&imageFeaturesAcc[0], &imageFeaturesAcc[1], &matchesAcc[0], rVecAB, tVecAB);
	EstimateCameraRotation(&imageFeaturesAcc[1], &imageFeaturesAcc[2], &matchesAcc[1], rVecBC, tVecBC);
	EstimateCameraRotation(&imageFeaturesAcc[2], &imageFeaturesAcc[3], &matchesAcc[2], rVecCD, tVecCD);
	EstimateCameraRotation(&imageFeaturesAcc[3], &imageFeaturesAcc[4], &matchesAcc[3], rVecDE, tVecDE);
	EstimateCameraRotation(&imageFeaturesAcc[4], &imageFeaturesAcc[0], &matchesAcc[4], rVecEA, tVecEA);
	rotationVec[0] = rVecAB;
	rotationVec[1] = rVecBC;
	rotationVec[2] = rVecCD;
	rotationVec[3] = rVecDE;
	rotationVec[4] = rVecEA;

	translationVec[0] = tVecAB;
	translationVec[1] = tVecBC;
	translationVec[2] = tVecCD;
	translationVec[3] = tVecDE;
	translationVec[4] = tVecEA;
}

void OmniStitcherR::FillInCameraParameters(void)
{
	/*
	cameras_.resize(NUM_OF_CAMERAS);
	const float degreeToRad = 3.14f / 360.0f;
	float angles[] = {0  * degreeToRad, +72  * degreeToRad, +144  * degreeToRad, -144  * degreeToRad, -72  * degreeToRad};
	//float angles[] = {-144 * degreeToRad, -72 * degreeToRad, 0 * degreeToRad, 72 * degreeToRad, 144 * degreeToRad};


	for (int i = 0; i < cameras_.size(); i++)
	{
		cameras_[i].focal = 1184.62;
		cameras_[i].aspect = 1;
		cameras_[i].ppx = 720;
		cameras_[i].ppy = 960;
		
		cv::Mat_<float> rotVec(3,1);
		rotVec.at<float>(0,0) = 0;
		rotVec.at<float>(1,0) = angles[i];
		rotVec.at<float>(2,0) = 0;
		cv::Mat rotMat;
		cv::Rodrigues(rotVec, rotMat);
		cameras_[i].R = rotMat;

		cv::Mat_<float> K;
		cameras_[i].K().convertTo(K, CV_32F);
		float test;
		test = K(0,0);
		test = K(0,2);
		test = K(1,1);
		test = K(1,2);

	}
	*/
}

void OmniStitcherR::EstimateCameraRotation(detail::ImageFeatures *featOne, detail::ImageFeatures *featTwo, std::vector<DMatch> *matches, cv::Mat &rVec, cv::Mat &tVec)
{
	vector<Point3f> srcPt;
	vector<Point2f> dstPt;

	srcPt.resize(matches->size());
	dstPt.resize(matches->size());

	for (int i = 0; i < matches->size(); i++)
	{
		srcPt[i] = Point3f( featOne->keypoints[matches->at(i).queryIdx].pt.x, featOne->keypoints[matches->at(i).queryIdx].pt.y,1);
		dstPt[i] = featTwo->keypoints[matches->at(i).trainIdx].pt;
	}
	solvePnPRansac(srcPt,dstPt, cameraIntrinsic_, cv::noArray(), rVec, tVec, false);
}

void OmniStitcherR::ReadFrame(std::string filename, std::vector<Mat> &images)
{
	assert(images.size() == NUM_OF_CAMERAS);
	for (int i = 0; i < folders_.size(); i++)
	{
		Mat img = imread(folders_[i] + filename);
		transpose(img,img);
		UndistortImage(img);
		img.copyTo(images.at(i));
	}
}

bool OmniStitcherR::IsReadyForStitching(void)
{
	return hasCalibrationData_ & hasInputFolderData_ & hasOutputFolderData_ & hasCameraParams_;
}

void OmniStitcherR::SetInputFolders(const std::vector<std::string> &folders)
{
	assert(folders.size()==NUM_OF_CAMERAS);
	for(auto &elem : folders)
		folders_.push_back(elem);
	hasInputFolderData_ = true;
}

void OmniStitcherR::SetOutputFolder(const std::string &folder)
{
	outFolder_ = folder;
	hasOutputFolderData_ = true;
}

bool OmniStitcherR::LoadCalibrationData(const std::string calibrationFilename)
{
	FileStorage fs(calibrationFilename,FileStorage::READ);
	if(fs.isOpened())
	{
		fs["cameraMatrix"] >> cameraIntrinsic_;
		fs["distCoeffs"] >> cameraDistortion_;
		fs.release();

		hasCalibrationData_ = true;
		return true;
	}
	return false;
}

void OmniStitcherR::UndistortImage(cv::Mat &frame)
{
	if(hasCalibrationData_)
	{
		if(!isCameraUndistortMapComputed_)
		{
			Mat cameraIntrinsicCalib_ = getOptimalNewCameraMatrix(cameraIntrinsic_, cameraDistortion_, frame.size(), 1.0);
			initUndistortRectifyMap(cameraIntrinsic_, cameraDistortion_, noArray(),  cameraIntrinsicCalib_, frame.size(), CV_32FC1, cameraUndistortMapX_, cameraUndistortMapY_); 
		}
		Mat outFrame;

		remap(frame, outFrame, cameraUndistortMapX_, cameraUndistortMapY_, INTER_LINEAR);
		swap(frame, outFrame);
	}
}

void OmniStitcherR::ExtractFeatures(Mat *frame, detail::ImageFeatures *features)
{
	detail::SurfFeaturesFinder surf;
	surf(*frame, *features);
}

void OmniStitcherR::AppendMatchingData(const std::vector<detail::ImageFeatures> &features, const std::vector<std::vector<DMatch>> &matches, std::vector<detail::ImageFeatures> &featuresAcc, std::vector<std::vector<DMatch>> &matchesAcc)
{
	std::vector<size_t> offset;
	for (int i = 0; i < featuresAcc.size(); i++)
	{
		offset.push_back(featuresAcc[i].keypoints.size());
	}

	for (int i = 0; i < features.size(); i++)
	{
		featuresAcc[i].descriptors.push_back(features[i].descriptors);
		for (int j = 0; j < features[i].keypoints.size(); j++)
		{
			featuresAcc[i].keypoints.push_back(features[i].keypoints[j]);
		}
	}

	for (int i = 0; i < matches.size(); i++)
	{
		size_t offsetQuery = offset[i];
		size_t offsetTrain = offset[ ((i<NUM_OF_CAMERAS-1) ? (i+1) : 0) ];
		for (int j = 0; j < matches[i].size(); j++)
		{
			DMatch match = matches[i][j];
			match.queryIdx += offsetQuery;
			match.trainIdx += offsetTrain;
			matchesAcc[i].push_back(match);
		}
	}
}


void OmniStitcherR::MatchFrameParallelFor(std::vector<cv::Mat> &images, std::vector<detail::ImageFeatures> &features, std::vector<std::vector<DMatch>> &matches)
{
	std::vector<detail::ImageFeatures> features_;
	parallel_for_(cv::Range(0,4),OmniMatchImage(&features_, &images));
}


void OmniStitcherR::MatchFrame(std::vector<Mat> &images, std::vector<detail::ImageFeatures> &features, std::vector<std::vector<DMatch>> &matches)
{
	/*std::vector<detail::ImageFeatures> features_;
	features_.resize(5);
	cv::setNumThreads(2);
	parallel_for_(cv::Range(0,4),OmniMatchImage(&features_, &images));*/
	detail::ImageFeatures featAt, featBt, featCt, featDt, featEt;
	std::vector<DMatch> matchesAB, matchesBC, matchesCD, matchesDE, matchesEA;

	std::thread extA(&OmniStitcherR::ExtractFeatures, this, &images[0], &featAt);
	std::thread extB(&OmniStitcherR::ExtractFeatures, this, &images[1], &featBt);
	std::thread extC(&OmniStitcherR::ExtractFeatures, this, &images[2], &featCt);
	std::thread extD(&OmniStitcherR::ExtractFeatures, this, &images[3], &featDt);
	std::thread extE(&OmniStitcherR::ExtractFeatures, this, &images[4], &featEt);
	extA.join();
	extB.join();
	std::thread matA(&OmniStitcherR::MatchImage, this, &featAt, &featBt, &matchesAB);
	extC.join();
	std::thread matB(&OmniStitcherR::MatchImage, this, &featBt, &featCt, &matchesBC);
	extD.join();
	std::thread matC(&OmniStitcherR::MatchImage, this, &featCt, &featDt, &matchesCD);
	extE.join();
	std::thread matD(&OmniStitcherR::MatchImage, this, &featDt, &featEt, &matchesDE);
	std::thread matE(&OmniStitcherR::MatchImage, this, &featEt, &featAt, &matchesEA);

	matA.join();
	matB.join();
	matC.join();
	matD.join();
	matE.join();

	features.push_back(featAt);
	features.push_back(featBt);
	features.push_back(featCt);
	features.push_back(featDt);
	features.push_back(featEt);

	matches.push_back(matchesAB);
	matches.push_back(matchesBC);
	matches.push_back(matchesCD);
	matches.push_back(matchesDE);
	matches.push_back(matchesEA);
}

void OmniStitcherR::MatchImage(detail::ImageFeatures *feature1, detail::ImageFeatures *feature2, std::vector<DMatch> *matches)
{
	FlannBasedMatcher matcher;
	std::vector<DMatch> matchesTemp;
	matcher.match(feature1->descriptors, feature2->descriptors, matchesTemp);
	double minDist = 100;
	double maxDist = 0;

	for (int i = 0; i < feature1->descriptors.rows; i++)
	{
		double dist = matchesTemp[i].distance;
		if(dist < minDist) minDist = dist;
		if(dist < maxDist) maxDist = dist;
	}

	for (int i = 0; i < feature1->descriptors.rows; i++)
	{
		if(matchesTemp[i].distance < 3 * minDist)
			matches->push_back(matchesTemp[i]);
	}
}

void OmniStitcherR::SaveCalibrationData(const std::string calibrationFilename)
{
	cv::FileStorage fs(calibrationFilename, cv::FileStorage::WRITE);
	if(!fs.isOpened())
	{
		std::cerr << "Cannot open file for writing\n";
		fs.release();
		return;
	}
	fs << "cameraMatrix" << cameraIntrinsic_;
	fs << "distCoeffs" << cameraDistortion_;

	fs << "rVecAB" << rotationVec[0];
	fs << "rVecBC" << rotationVec[1];
	fs << "rVecCD" << rotationVec[2];
	fs << "rVecDE" << rotationVec[3];
	fs << "rVecEA" << rotationVec[4];

	fs << "tVecAB" << translationVec[0];
	fs << "tVecBC" << translationVec[1];
	fs << "tVecCD" << translationVec[2];
	fs << "tVecDE" << translationVec[3];
	fs << "tVecEA" << translationVec[4];

	fs.release();
}

void OmniStitcherR::PerformCameraCalibration(const std::vector<std::string> &images, const int width, const int height, const float physWidth, const float physHeight)
{
	std::vector<std::vector<cv::Point2f>> imgPoints;
	std::vector<std::vector<cv::Point3f>> objPoints;
	std::vector<cv::Point3f> objPointsInstance;
	for(int i=0; i<height; ++i){
		for(int j=0; j<width; ++j){
			objPointsInstance.push_back(cv::Point3f(j * physWidth, i * physHeight, 0));
		}
	}
	imgPoints.resize(images.size());
	objPoints.resize(images.size());
	cv::Size gridSize(width,height);
	cv::setNumThreads(4);
	parallel_for_(cv::Range(0,images.size()), OmniCamCalib(&images, &objPointsInstance, &gridSize, &imgPoints, &objPoints));
	std::vector<int> mask;
	int i = 0;
	for(auto itr = imgPoints.cbegin(); itr!= imgPoints.cend(); ++itr, ++i){
		if(itr->empty())
			mask.push_back(i);
	}
	for(int i = mask.size()-1; i > -1; --i){
		imgPoints.erase(imgPoints.begin()+i);
		objPoints.erase(objPoints.begin()+i);
	}
	cv::Size imgSize;
	cv::Mat img = cv::imread(images[0]);
	imgSize = img.size();
	img.release();
	double result = cv::calibrateCamera(objPoints, imgPoints, imgSize, cameraIntrinsic_, cameraDistortion_, cv::noArray(), cv::noArray());
	hasCalibrationData_ = true;
}