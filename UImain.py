import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import *
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from SearchPaper import myredect
from mainwindows import Ui_MainWindow


class Search(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Search, self).__init__(parent)
        self.setupUi(self)

        self.pushButton.clicked.connect(self.search)

    def search(self):
        sentence, scores = myredect(self.textEdit.toPlainText())
        self.listWidget.clear()
        for i in range(len(sentence)):
            self.listWidget.addItem("top " + str(i + 1) + ":" + sentence[i])
            self.listWidget.addItem("         similarity is :" + str(scores[i]))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Please enter the title"))
        self.pushButton.setText(_translate("MainWindow", "submit"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Search()
    ui.show()
    sys.exit(app.exec_())
