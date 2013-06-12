#pragma once
using <string>

class ParamReader
{
public:
	ParamReader(){}
	~ParamReader(){}
	
	loadParamFile(std::string filename);
private:
	checkParamFile();
	std::vector<std::string> sourceFolders;
	std::string calibFilename;
	std::string outputDirectory;
};
