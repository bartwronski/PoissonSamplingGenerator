from __future__ import unicode_literals
import sys
from poisson_ui import Ui_MainWindow
import poisson

from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.figure = Figure(figsize=(5, 5), dpi=100)
        self.ui.plot_canvas = FigureCanvas(self.ui.figure)
        self.ui.plot_canvas.setParent(self.ui.widget)
        
        self.ui.plot_canvas.setSizePolicy(
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        self.ui.plot_canvas.updateGeometry()

        self.ui.generateButton.clicked.connect(self.generate)
        self.task = TaskThread()
        self.task.taskFinished.connect(self.onFinished)
        self.task.notifyProgress.connect(self.onProgress)
        self.taskStarted = False

    def generate(self):
        if self.taskStarted == True:
            return

        self.taskStarted = True
        disk = False        
        repeatPattern = True
        num_dim = 2         
        allow_rotations = False

        if self.ui.radioButton1DLine.isChecked():
            disk = False
            num_dim = 1
            repeatPattern = False
        elif self.ui.radioButton1DRepeatedLine.isChecked():
            disk = False
            num_dim = 1
            repeatPattern = True
        elif self.ui.radioButton2DDisk.isChecked():
            disk = True
            num_dim = 2
            repeatPattern = False
        elif self.ui.radioButton2DRotatedDisk.isChecked():
            disk = True
            num_dim = 2
            repeatPattern = False
            allow_rotations = True
        elif self.ui.radioButton2DRect.isChecked():
            disk = False
            num_dim = 2
            repeatPattern = False
        elif self.ui.radioButton2DRepeatedRectangle.isChecked():
            disk = False
            num_dim = 2
            repeatPattern = True
        elif self.ui.radioButton3DSphere.isChecked():
            disk = True
            num_dim = 3
            repeatPattern = False
        elif self.ui.radioButton3DBox.isChecked():
            disk = False
            num_dim = 3
            repeatPattern = False
        elif self.ui.radioButton3DRepeatedBox.isChecked():
            disk = False
            num_dim = 3
            repeatPattern = True

        # user defined options
        num_points = self.ui.numberOfPointsSpinBox.value()
        num_iterations = self.ui.numberTotalIterationsSpinBox.value()
        first_point_zero = not self.ui.firstPointRandomCheckBox.isChecked()
        iterations_per_point = self.ui.numberOfIterationsPerPointSpinBox.value()
        sorting_buckets = self.ui.cacheSortBucketsSpinBox.value()
        rotations = self.ui.rotationsAsRepetitions.value() if allow_rotations else 1

        poisson_generator = poisson.PoissonGenerator(num_dim, disk, repeatPattern, first_point_zero)
        self.task.poisson_generator = poisson_generator
        self.task.num_points = num_points
        self.task.num_iterations = num_iterations
        self.task.iterations_per_point = iterations_per_point
        self.task.sorting_buckets = sorting_buckets
        self.task.rotations = rotations
        self.task.start()

    def onFinished(self):
        points = self.task.points
        points = self.task.poisson_generator.cache_sort(points, self.task.sorting_buckets)
        text_output = self.task.poisson_generator.format_points_string(points)
        print(text_output)
        self.ui.outputShaderCodeTextEdit.clear()
        self.ui.outputShaderCodeTextEdit.insertPlainText(text_output)
        self.ui.figure.clear()
        self.task.poisson_generator.generate_ui(self.ui.figure, points)
        self.ui.plot_canvas.updateGeometry()
        self.ui.plot_canvas.draw()
        self.ui.progressBar.setValue(1024)
        self.taskStarted = False

    def onProgress(self, i):
        self.ui.progressBar.setValue(int(round(i * 1024)))

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

class TaskThread(QtCore.QThread):
    taskFinished = QtCore.pyqtSignal()
    notifyProgress = QtCore.pyqtSignal(float)
    def run(self):
        self.points = self.poisson_generator.find_point_set(self.num_points, self.num_iterations, self.iterations_per_point, self.rotations, self.progress_fun)
        self.taskFinished.emit() 

    def progress_fun(self, val):
        self.notifyProgress.emit(val)

if __name__ == '__main__':
    qApp = QtGui.QApplication(sys.argv)

    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
