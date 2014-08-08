#include "facerec.h"
#include <QtGui/QApplication>

#include <windows.h>

int main(int argc, char *argv[])
{
	AllocConsole();
	//freopen("test1.txt","w",stdout);
	freopen("CONOUT$","w",stdout);

	QApplication a(argc, argv);
	facerec w;
	w.show();
	return a.exec();
}
