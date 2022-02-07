
#include <string>
using namespace std;
class CunetaFolderManager
{
public:
	int totalFolders;
	int currentFolder;
	string dataDirectory;
	CunetaFolderManager(string dataDirectory);
	void GetAllFoldersInDirectory();
	void GetNextFolder();
};