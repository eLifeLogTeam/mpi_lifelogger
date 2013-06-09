#pragma once
#include <vector>
#include <opencv2\opencv.hpp>
using namespace cv;



class OmniStitcherR
{
public:
	OmniStitcherR(void);
	~OmniStitcherR(void);

	void SetInputFolders(const std::vector<std::string> &folders);
	void SetOutputFolder(const std::string &folder);
	bool LoadCalibrationData(const std::string calibrationFilename);
	void SaveCalibrationData(const std::string calibrationFilename);
	void EstimateCameraParameters(const std::vector<std::string> &testFiles);
	void StitchFrame(std::string filename);
	void FillInCameraParameters(void);
	void MatchFrameParallelFor(std::vector<cv::Mat> &images, std::vector<detail::ImageFeatures> &features, std::vector<std::vector<DMatch>> &matches);

private:
	void UndistortImage(cv::Mat &frame);
	void ReadFrame(std::string filename, std::vector<Mat> &images);
	void ExtractFeatures(Mat *frame, detail::ImageFeatures *features);
	void MatchFrame(std::vector<Mat> &images, std::vector<detail::ImageFeatures> &features, std::vector<std::vector<DMatch>> &matches);
	void MatchFrameParallelFor(std::vector<std::string> &images, std::vector<detail::ImageFeatures> &features, std::vector<std::vector<DMatch>> &matches);
	void MatchImage(detail::ImageFeatures *feature1, detail::ImageFeatures *feature2, std::vector<DMatch> *matches);
	void AppendMatchingData(const std::vector<detail::ImageFeatures> &features, const std::vector<std::vector<DMatch>> &matches, std::vector<detail::ImageFeatures> &featuresAcc, std::vector<std::vector<DMatch>> &matchesAcc);
	void EstimateCameraRotation(detail::ImageFeatures *featOne, detail::ImageFeatures *featTwo, std::vector<DMatch> *matches, cv::Mat &rVec, cv::Mat &tVec);
	bool IsReadyForStitching(void);
	
	double workScale_;

	const static int NUM_OF_CAMERAS = 5;
	const std::string CAMERA_MATRIX_IDEN;
	const std::string DIST_COEFF_IDEN;

	std::vector<cv::Mat> rotationVec;
	std::vector<cv::Mat> translationVec;


	bool hasInputFolderData_;
	bool hasOutputFolderData_;
	bool hasCameraParams_;
	bool hasCalibrationData_;
	bool isCameraUndistortMapComputed_;

	std::vector<std::string> folders_;
	std::string outFolder_;

	
	Mat cameraIntrinsic_;
	Mat cameraDistortion_;
	Mat cameraUndistortMapX_;
	Mat cameraUndistortMapY_;
};

class OmniMatchImage : public cv::ParallelLoopBody
{
public:
	OmniMatchImage(std::vector<detail::ImageFeatures> *features, std::vector<cv::Mat> *images)
		:features_(features), images_(images) {}
	void operator()(const cv::Range &r) const
	{
		for(int ji = r.start; ji != r.end; ++ji)
		{
			detail::SurfFeaturesFinder surf;
			surf(images_->at(ji), features_->at(ji));
		}
	}

private:
	std::vector<detail::ImageFeatures> *features_;
	std::vector<cv::Mat> *images_;

};

