# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BookPageRangeDialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(336, 102)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label_total_pages = QtWidgets.QLabel(Dialog)
        self.label_total_pages.setObjectName("label_total_pages")
        self.gridLayout.addWidget(self.label_total_pages, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.spinBox_from = QtWidgets.QSpinBox(Dialog)
        self.spinBox_from.setMinimum(1)
        self.spinBox_from.setObjectName("spinBox_from")
        self.gridLayout.addWidget(self.spinBox_from, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 2, 1, 1)
        self.spinBox_to = QtWidgets.QSpinBox(Dialog)
        self.spinBox_to.setMinimum(1)
        self.spinBox_to.setObjectName("spinBox_to")
        self.gridLayout.addWidget(self.spinBox_to, 1, 3, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 4)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        self.spinBox_to.valueChanged['int'].connect(Dialog.to_changed) # type: ignore
        self.spinBox_from.valueChanged['int'].connect(Dialog.from_changed) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_total_pages.setText(_translate("Dialog", "Total Pages:"))
        self.label_2.setText(_translate("Dialog", "From:"))
        self.label_3.setText(_translate("Dialog", "To:"))
