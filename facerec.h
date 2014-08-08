#ifndef FACEREC_H
#define FACEREC_H

#include <QtGui/QMainWindow>
#include "ui_facerec.h"

class facerec : public QMainWindow
{
	Q_OBJECT

public:
	facerec(QWidget *parent = 0, Qt::WFlags flags = 0);
	~facerec();

private slots:
	void onBtnAddSampleClicked();
	void onBtnTrainClicked();
	void onBtnRecCamClicked();
private:
	Ui::facerecClass ui;
};

#endif // FACEREC_H
