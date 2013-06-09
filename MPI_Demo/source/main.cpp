// OmniVideoSticher.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include "OmniStitcherR.h"

using namespace cv;


void TestRefactored(void)
{
	OmniStitcherR stitcher;
	std::string fA = "c:/Users/Oleg/Videos/OmniVideo_Prototype/Renders/Camera_Alfa/";
	std::string fB = "c:/Users/Oleg/Videos/OmniVideo_Prototype/Renders/Camera_Bravo/";
	std::string fC = "c:/Users/Oleg/Videos/OmniVideo_Prototype/Renders/Camera_Charlie/";
	std::string fD = "c:/Users/Oleg/Videos/OmniVideo_Prototype/Renders/Camera_Delta/";
	std::string fE = "c:/Users/Oleg/Videos/OmniVideo_Prototype/Renders/Camera_Echo/";
	std::string calibrationFilename = "c:/Users/Oleg/Desktop/calibData.txt";
	std::vector<std::string> inputFolders;
	inputFolders.push_back(fA);
	inputFolders.push_back(fB);
	inputFolders.push_back(fC);
	inputFolders.push_back(fD);
	inputFolders.push_back(fE);

	stitcher.SetInputFolders(inputFolders);
	stitcher.LoadCalibrationData(calibrationFilename);

	std::vector<std::string> files;
	files.push_back("frames_000003.jpeg");
	//files.push_back("frames_000250.jpeg");
	/*files.push_back("frames_000500.jpeg");
	files.push_back("frames_000750.jpeg");
	files.push_back("frames_001000.jpeg");
	files.push_back("frames_001250.jpeg");
	files.push_back("frames_001500.jpeg");
	files.push_back("frames_001750.jpeg");
	files.push_back("frames_002000.jpeg");
	files.push_back("frames_002250.jpeg");
	files.push_back("frames_002500.jpeg");
	files.push_back("frames_002750.jpeg");*/
	stitcher.EstimateCameraParameters(files);
}


int _tmain(int argc, _TCHAR* argv[])
{
	TestRefactored();

	return 0;
}

