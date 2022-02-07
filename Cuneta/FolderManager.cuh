
#include <string>
#include <vector>
using namespace std;
class CunetaFolderManager
{
public:
	int totalFolders;
	int currentFolderIndex;
	string currentFolder;
	string currentImageName;
	string dataDirectory;
	vector<std::string> directoryList;
	
	CunetaFolderManager(string _dataDirectory);
	void GetAllFoldersInDirectory();
	void OpenNextFolder();
	void UpdateImageName();
};