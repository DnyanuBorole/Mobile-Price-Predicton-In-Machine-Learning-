# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\QT\Cancer\cancer.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import numpy as np
from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
from call import predict
# from Graphs import view
from Classification import classfy
import csv
import sys





class Ui_Dialog(object):

    def clear(self):
        self.lineEdit.clear()
        self.lineEdit_2.clear()
        self.lineEdit_3.clear()
        self.lineEdit_4.clear()
        self.lineEdit_5.c   lear()
        self.lineEdit_6.clear()
        self.lineEdit_7.clear()
        self.lineEdit_8.clear()
        self.lineEdit_9.clear()
        self.lineEdit_10.clear()
        self.lineEdit_11.clear()
        self.lineEdit_12.clear()

        # read_data = file.read()
    def alertmsg(self, title, Message):
        warn = QtWidgets.QMessageBox()
        warn.setIcon(QtWidgets.QMessageBox.Information)
        warn.setWindowTitle(title)
        warn.setText(Message)
        warn.setStandardButtons(QtWidgets.QMessageBox.Ok)
        warn.exec_()

    def prediction(self):
        v = self.lineEdit.text()
        v1 = self.lineEdit_2.text()
        self.lineEdit_2.setValidator(QtGui.QIntValidator())
        v2 = self.lineEdit_3.text()
        self.lineEdit_3.setValidator(QtGui.QIntValidator())
        v3 = self.lineEdit_4.text()
        self.lineEdit_4.setValidator(QtGui.QIntValidator())
        v4 = self.lineEdit_5.text()
        self.lineEdit_5.setValidator(QtGui.QIntValidator())
        v5 = self.lineEdit_6.text()
        self.lineEdit_6.setValidator(QtGui.QIntValidator())
        v6 = self.lineEdit_7.text()
        self.lineEdit_7.setValidator(QtGui.QIntValidator())
        v7 = self.lineEdit_8.text()
        self.lineEdit_8.setValidator(QtGui.QIntValidator())
        v8 = self.lineEdit_9.text()
        self.lineEdit_9.setValidator(QtGui.QIntValidator())
        v9 = self.lineEdit_10.text()
        self.lineEdit_10.setValidator(QtGui.QIntValidator())
        v10 = self.lineEdit_11.text()
        self.lineEdit_11.setValidator(QtGui.QIntValidator())
        v11 = self.lineEdit_12.text()
        self.lineEdit_12.setValidator(QtGui.QIntValidator())
        v12 = self.comboBox.currentText()

        if v == '' or v == "null" or v1 == '' or v1 == "null" or v2 == '' or v2 == "null" or v3 == '' or v3 == "null" or v4 == '' or v4 == "null" or v5 == '' or v5 == "null" or v6 == '' or v6 == "null" or v7 == '' or v7 == "null" or v8 == '' or v8 == "null" or v9 == '' or v9 == "null" or v10 == '' or v10 == "null" or v11 == '' or v11 == "null" or v12 == '' or v12 == "null":
            self.alertmsg("error", "Please fill the All Details")
        else:
            if v12 == 'Male':
                v12 = '1'
            else:
                v12 = '0'

        fields = ('Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'Balanced Diet', 'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Weight Loss', 'Dry Cough', 'Snoring')
        rows = [[v, v12, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11]]

        with open('test.csv', 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)

        print("Writing complete")
        result = predict("trained.csv", "test.csv")
        print(result)

        self.output.setText(result)

    def setupUi(self, Dialogui):
        Dialogui.setObjectName("Dialog")
        Dialogui.resize(778, 831)
        # Dialog.setStyleSheet("QDialog{\n""    background-image: url(:\Machine learning\CancerPred\image1.jpg);}")
        self.label_2.setGeometry(QtCore.QRect(0, 0, 871, 51))

        self.label_2.setStyleSheet("QLabel{\n""\n""background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, "
                                   "y2:0, stop:0 rgba(255, 178, 102, 255), stop:0.55 rgba(235, 148, 61, 255), "
                                   "stop:0.98 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));}")
        self.label_2.setObjectName("label_2")
        self.label.setGeometry(QtCore.QRect(222, 170, 61, 21))
        self.label.setStyleSheet("font: 11pt \"Algerian\";")
        self.label.setObjectName("label")
        self.lineEdit.setGeometry(QtCore.QRect(280, 170, 61, 22))
        self.lineEdit.setObjectName("lineEdit")
        self.comboBox.setGeometry(QtCore.QRect(460, 170, 111, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setCurrentIndex(-1)
        self.label_3.setGeometry(QtCore.QRect(370, 170, 71, 21))
        self.label_3.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_3.setObjectName("label_3")
        self.label_4.setGeometry(QtCore.QRect(50, 230, 131, 31))
        self.label_4.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_4.setObjectName("label_4")
        self.label_5.setGeometry(QtCore.QRect(40, 270, 141, 31))
        self.label_5.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_5.setObjectName("label_5")
        self.label_6.setGeometry(QtCore.QRect(50, 320, 131, 21))
        self.label_6.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_6.setObjectName("label_6")
        self.label_7.setGeometry(QtCore.QRect(40, 370, 141, 21))
        self.label_7.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_7.setObjectName("label_7")
        self.label_8.setGeometry(QtCore.QRect(98, 420, 91, 21))
        self.label_8.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_8.setObjectName("label_8")
        self.label_9.setGeometry(QtCore.QRect(20, 470, 151, 31))
        self.label_9.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_9.setObjectName("label_9")
        self.label_10.setGeometry(QtCore.QRect(470, 260, 101, 31))
        self.label_10.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_10.setObjectName("label_10")
        self.label_11.setGeometry(QtCore.QRect(400, 300, 171, 41))
        self.label_11.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_11.setObjectName("label_11")
        self.label_12.setGeometry(QtCore.QRect(460, 360, 111, 21))
        self.label_12.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_12.setObjectName("label_12")
        self.label_13.setGeometry(QtCore.QRect(470, 410, 101, 31))
        self.label_13.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_13.setObjectName("label_13")
        self.label_14.setGeometry(QtCore.QRect(500, 450, 71, 31))
        self.label_14.setStyleSheet("font: 11pt \"Algerian\";")
        self.label_14.setObjectName("label_14")
        self.lineEdit_2.setGeometry(QtCore.QRect(217, 236, 113, 22))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3.setGeometry(QtCore.QRect(217, 276, 113, 22))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4.setGeometry(QtCore.QRect(217, 320, 113, 22))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_5.setGeometry(QtCore.QRect(217, 370, 113, 22))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_6.setGeometry(QtCore.QRect(217, 420, 113, 22))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_7.setGeometry(QtCore.QRect(217, 470, 113, 22))
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.lineEdit_8.setGeometry(QtCore.QRect(600, 264, 113, 22))
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.lineEdit_9.setGeometry(QtCore.QRect(600, 307, 113, 22))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.lineEdit_10.setGeometry(QtCore.QRect(600, 358, 113, 22))
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.lineEdit_11.setGeometry(QtCore.QRect(600, 413, 113, 22))
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.lineEdit_12.setGeometry(QtCore.QRect(600, 454, 113, 22))
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.label_15.setGeometry(QtCore.QRect(110, 600, 131, 91))
        self.label_15.setStyleSheet("font: 75 22pt \"Rockwell Condensed\";\n""color: rgb(0, 85, 0);")
        self.label_15.setObjectName("label_15")
        self.output.setGeometry(QtCore.QRect(250, 600, 241, 91))
        self.output.setStyleSheet("font: 75 22pt \"Rockwell Condensed\";\n""color: rgb(255, 85, 127);")
        self.output.setText(" ")
        self.output.setObjectName("output")
        self.pushButton.setGeometry(QtCore.QRect(310, 530, 120, 35))
        self.pushButton.setStyleSheet("background-color: rgb(105, 200, 129);\n"

                                      "font: 87 16pt \"Arial Black\";\n"
                                      "text-decoration: underline;\n"
                                      "color: rgb(32, 29, 85);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2.setGeometry(QtCore.QRect(450, 530, 120, 35))
        self.pushButton_2.setStyleSheet(
            "background-color: rgb(105, 200, 129);\n""font: 87 12pt \"Arial Black\";\n""text-decoration: underline;\n"
            "color: rgb(32, 29, 85);")

        self.retranslateUi(Dialogui)
        QtCore.QMetaObject.connectSlotsByName(Dialogui)

        self.pushButton_2.clicked.connect(self.clear)
        self.pushButton.clicked.connect(self.prediction)

    def __init__(self):
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.output = QtWidgets.QLabel(Dialog)
        self.label_15 = QtWidgets.QLabel(Dialog)
        self.lineEdit_12 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_11 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_10 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_9 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_8 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_7 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_6 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_5 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_4 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_3 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_2 = QtWidgets.QLineEdit(Dialog)
        self.label_14 = QtWidgets.QLabel(Dialog)
        self.label_13 = QtWidgets.QLabel(Dialog)
        self.label_12 = QtWidgets.QLabel(Dialog)
        self.label_11 = QtWidgets.QLabel(Dialog)
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.label = QtWidgets.QLabel(Dialog)

    def retranslateUi(self, Dialogui):
        _translate = QtCore.QCoreApplication.translate
        Dialogui.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_2.setText(_translate("Dialog",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-size:26pt; "
                                        "font-weight:600; color:#ffffff;\">CANCER "
                                        "PREDICTION</span></p></body></html>"))
        self.label.setText(_translate("Dialog", "AGE"))
        self.comboBox.setItemText(0, _translate("Dialog", "Male"))
        self.comboBox.setItemText(1, _translate("Dialog", "Female"))
        self.label_3.setText(_translate("Dialog", "Gender"))
        self.label_4.setText(_translate("Dialog", "Air Pollution"))
        self.label_5.setText(_translate("Dialog", "Drink Alcohol"))
        self.label_6.setText(_translate("Dialog", "Dust Allergy"))
        self.label_7.setText(_translate("Dialog", "Balanced Diet"))
        self.label_8.setText(_translate("Dialog", "Smoking"))
        self.label_9.setText(_translate("Dialog", "Passive Smoking"))
        self.label_10.setText(_translate("Dialog", "Chest Pain"))
        self.label_11.setText(_translate("Dialog", "Coughing Of Blood"))
        self.label_12.setText(_translate("Dialog", "Weight Loss"))
        self.label_13.setText(_translate("Dialog", "Dry Cough"))
        self.label_14.setText(_translate("Dialog", "Snoring"))
        self.label_15.setText(_translate("Dialog", "RESULT :"))
        self.pushButton.setText(_translate("Dialog", "Predict"))
        self.pushButton_2.setText(_translate("Dialog", "Clear All"))


if __name__ == "__main__":
    # import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
