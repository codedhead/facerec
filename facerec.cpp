#include "facerec.h"
#include <qmessagebox>
#include <qfiledialog>
#include <QStringList>
#include <QSTRING>
#include <QListView>
#include <QTreeView>
#include <QDir>
#include <QProcess>
#include <QComboBox>

#include <windows.h>

#include "cvface.h"

using namespace std;
using namespace cvface;

facerec::facerec(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);
}

facerec::~facerec()
{

}

#define DATA_PATH "data/"

void showInGraphicalShell(const QString &filePath)
{
#if defined(Q_OS_WIN)
// 	const QString explorer = Environment::systemEnvironment().searchInPath(QLatin1String("explorer.exe"));
// 	if (explorer.isEmpty()) {
// 		QMessageBox::warning(NULL,
// 			tr("Launching Windows Explorer failed"),
// 			tr("Could not find explorer.exe in path to launch Windows Explorer."));
// 		return;
// 	}
// 	QString param;
// 	if (!QFileInfo(pathIn).isDir())
// 		param = QLatin1String("/select,");
// 	param += QDir::toNativeSeparators(pathIn);
// 	QProcess::startDetached(explorer, QStringList(param));
	QStringList args;
	args << "/select," << QDir::toNativeSeparators(filePath);
	QProcess::startDetached("explorer", args);

#endif

}

void preprocess(const QString& path)
{
	QDir dir(path);
	if(!dir.exists()) return;

	QString abs_path=dir.absolutePath()+'/';
	QString res_path=DATA_PATH+dir.dirName()+'/';
	QDir cur_dir;
	if(!cur_dir.exists(res_path))
		cur_dir.mkpath(res_path);

	QStringList filters;
	filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp" << "*.pgm";
	dir.setNameFilters(filters);
	dir.setFilter(QDir::Files);
	QStringList imagelist=dir.entryList();

	printf("dst_path: %s\n",res_path.toStdString().c_str());
	char fname[MAX_PATH],dst_fname[MAX_PATH];
	int good_i=-1;
	int succ_cnt=1;
	for(int i=0;i<imagelist.size();++i)
	{
		strcpy(fname,(abs_path+imagelist[i]).toStdString().c_str());
		//strcpy(dst_fname,(res_path+imagelist[i]).toStdString().c_str());
		QFileInfo finfo(fname);
		strcpy(dst_fname,(res_path+QString::number(succ_cnt)+"."+finfo.suffix()).toStdString().c_str());

		printf("%s\n",fname);
		if(cvface::normalizeSample(fname,dst_fname))
		{
			++succ_cnt;
			if(good_i==-1) good_i=i;
		}
	}
	if(good_i>=0)
	{
		strcpy(fname,(abs_path+imagelist[good_i]).toStdString().c_str());

		QString tmp_avatar_file="avatar."+imagelist[good_i];

		QString avatar_path=res_path+tmp_avatar_file;
		strcpy(dst_fname,avatar_path.toStdString().c_str());
		cvface::saveAvatar(fname,dst_fname);

		
		if(cur_dir.exists(avatar_path))
		{
			QDir res_dir(res_path);
			QString ava_file=dir.dirName()+".avatar";
			res_dir.remove(ava_file);
			res_dir.rename(tmp_avatar_file,ava_file);
		}

		//showInGraphicalShell(res_path);
	}
	printf("done\n");
}


void facerec::onBtnAddSampleClicked()
{
	QFileDialog dialog;
	dialog.setFileMode(QFileDialog::Directory);
	dialog.setOption(QFileDialog::ShowDirsOnly);
	QListView *l = dialog.findChild<QListView*>("listView");
	if (l) {
		l->setSelectionMode(QAbstractItemView::MultiSelection);
	}
	QTreeView *t = dialog.findChild<QTreeView*>();
	if (t) {
		t->setSelectionMode(QAbstractItemView::MultiSelection);
	}

	if(dialog.exec())
	{
		QStringList& folders=dialog.selectedFiles();
		for(int i=0;i<folders.size();++i)
		{
			printf("%s\n",folders[i].toStdString().c_str());
			preprocess(folders[i]);
		}
	}
}

#define FISHER_MODEL_FILE DATA_PATH "trained.fisher"
#define LBPH_MODEL_FILE DATA_PATH "trained.lbph"
char fisher_model_file[]=FISHER_MODEL_FILE;
char lbph_model_file[]=LBPH_MODEL_FILE;

void facerec::onBtnTrainClicked()
{
	int model_type=this->ui.comboType->currentIndex();
	char* model_file=0;
	if(model_type==0)
		model_file=fisher_model_file;
	else
		model_file=lbph_model_file;


	QString data_path=DATA_PATH;
	QDir data_dir(data_path);
	if(!data_dir.exists())
	{
		printf("Training data not found\n");
		return;
	}

	data_dir.setFilter(QDir::AllDirs|QDir::NoDotAndDotDot);
	QStringList dirlist=data_dir.entryList();


	QStringList image_filters;
	image_filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp" << "*.pgm";	


#ifdef LOOCV_TEST
	int loocv_i=0;
	int loocv_correct=0;
	QString loocv_fname;
	
	for(;;++loocv_i)
	{
		int loocv_label=-1;
#endif


	g_trainer.reset();
	int _idx=0; // counting

	char class_name[256],image_fname[MAX_PATH];
	for(int i=0;i<dirlist.size();++i)
	{
		strcpy(class_name,dirlist[i].toStdString().c_str());
		g_trainer.enterClass(class_name);

		QString class_path=data_path+dirlist[i]+'/';
		QDir class_dir(class_path);
		assert(class_dir.exists());

		class_dir.setNameFilters(image_filters);
		class_dir.setFilter(QDir::Files);
		QStringList imagelist=class_dir.entryList();
		for(int j=0;j<imagelist.size();++j,++_idx)
		{
#ifdef LOOCV_TEST
			if(_idx==loocv_i)
			{
				loocv_label=i;
				loocv_fname=class_path+imagelist[j];
				printf("DEBUG: train ignore, LOOCV %d\n",_idx);
				continue;
			}
#endif

			strcpy(image_fname,(class_path+imagelist[j]).toStdString().c_str());

			g_trainer.addSample(image_fname);
		}
	}

	
	g_trainer.train(model_file,model_type);

#ifdef LOOCV_TEST
	if(loocv_label==-1) break;
	printf("test: %s\n",loocv_fname.toStdString().c_str());
	if(loocvTest(model_file,model_type,loocv_fname.toStdString().c_str(),loocv_label))
		++loocv_correct;
	}
	printf("loocv_correct: %d\n",loocv_correct);
#endif
}

void facerec::onBtnRecCamClicked()
{
	int model_type=this->ui.comboType->currentIndex();
	char* model_file=0;
	if(model_type==0)
		model_file=fisher_model_file;
	else
		model_file=lbph_model_file;

	QDir dir;
	if(!dir.exists(model_file))
	{
		printf("Trained model file \"%s\" not found, please re-train the model\n",model_file);
	}
	else
	{
		doCapture(model_file,model_type);
	}
}