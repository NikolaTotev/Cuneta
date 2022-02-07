#include "FolderManager.cuh"
#include  <filesystem>
#include <iostream>
#ifndef _FILESYSTEM_;
#define  _FILESYSTEM_;
#endif

CunetaFolderManager::CunetaFolderManager(string _dataDirectory)
{
	dataDirectory = _dataDirectory;
}

std::vector<std::string> splitString(std::string str, char splitter) {
	std::vector<std::string> result;
	std::string current = "";
	for (int i = 0; i < str.size(); i++) {
		if (str[i] == splitter) {
			if (current != "") {
				result.push_back(current);
				current = "";
			}
			continue;
		}
		current += str[i];
	}
	if (current.size() != 0)
		result.push_back(current);
	return result;
}

void CunetaFolderManager::GetAllFoldersInDirectory()
{
	for (auto& p : std::filesystem::recursive_directory_iterator(dataDirectory))
	{
		if (p.is_directory())
		{
			directoryList.push_back(p.path().string());
		}
	}

	currentFolderIndex = 0;
	UpdateImageName();
}

void CunetaFolderManager::UpdateImageName()
{
	totalFolders = directoryList.size();
	currentFolder = directoryList[currentFolderIndex];
	vector<string> names = splitString(directoryList[currentFolderIndex], '\\');
	currentImageName = names[names.size() - 1];
}

void CunetaFolderManager::OpenNextFolder()
{
	if(currentFolderIndex < totalFolders-1)
	{
		currentFolderIndex++;
		UpdateImageName();
	}
	else
	{
		cout << "===========================" << endl;
		cout << ">>>> PROCESSING UPDATE <<<<" << endl;
		cout << "===========================" << endl;
		cout << "All folders in: " << endl;
		cout << dataDirectory << endl;
		cout << "have been processed." << endl;
	}
	
}

