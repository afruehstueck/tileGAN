from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import QObject, QPoint, QPointF, QFile, QSize, QSizeF, QRect, QRectF, QMimeData, Signal, Slot
from PIL import Image
import qdarkstyle
from multiprocessing.managers import BaseManager
import numpy as np
from pathlib import Path
import ctypes
import time
import colorsys
import os
from enum import Enum

USE_DARK_THEME = True

if USE_DARK_THEME:
	os.environ['QT_API'] = 'pyqt'
	iconFolder = 'icons/dark'
	styleColor = (20, 140, 210) #'148cd2'
else:
	iconFolder = 'icons/light'
	styleColor = (138, 198, 64) #'8ac546' #green

#this is necessary in order to display the correct taskbar icon
myappid = 'application.tileGAN' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class Mode(Enum):
	NONE = 0,
	MOVE = 1,
	RESIZETL = 2,
	RESIZET = 3,
	RESIZETR = 4,
	RESIZER = 5,
	RESIZEBR = 6,
	RESIZEB = 7,
	RESIZEBL = 8,
	RESIZEL = 9

class FloatViewer(QWidget):
	""" Widget that can be moved and resized by user"""
	menu = None
	mode = Mode.NONE
	position = None
	focusIn = Signal(bool)
	focusOut = Signal(bool)
	newGeometry = Signal(QRect)

	def __init__(self, parent, p, preserveAspectRatio=False):
		super().__init__(parent=parent)

		self.menu = QMenu(parent=self, title='menu')
		self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
		self.setVisible(True)
		self.setAutoFillBackground(False)
		self.setMouseTracking(True)
		self.setFocusPolicy(QtCore.Qt.ClickFocus)
		self.setFocus()
		self.move(p)
		self.imageViewerLabel = QLabel()
		self.pixmap = None
		self.aspectRatio = 1
		self.childWidget = None

		self.vLayout = QVBoxLayout(self)
		self.setChildWidget(self.imageViewerLabel)

		self._inFocus = True
		self._showMenu = False
		self._isEditing = True
		self._preserveAspectRatio = preserveAspectRatio
		self.installEventFilter(parent)

	def hide(self):
		self.setVisible(False)

	def show(self):
		self.setVisible(True)

	def toggle(self):
		self.setVisible(not self.isVisible())

	def setImage(self, pixmap):
		self.pixmap = pixmap
		self.aspectRatio = self.pixmap.height()/self.pixmap.width()
		self.imageViewerLabel.setPixmap(self.pixmap.scaledToHeight(128))
		self.adjustSize()

	def resizeImage(self):
		pass

	def setChildWidget(self, cWidget):
		if cWidget:
			self.childWidget = cWidget
			self.childWidget.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
			self.childWidget.setParent(self)
			self.childWidget.releaseMouse()
			self.vLayout.addWidget(cWidget)
			self.vLayout.setContentsMargins(0, 0, 0, 0)
			self.adjustSize()

	def popupShow(self, pt: QPoint):
		if self.menu.isEmpty:
			return
		global_ = self.mapToGlobal(pt)
		self._showMenu = True
		self.menu.exec(global_)
		self._showMenu = False

	def focusInEvent(self, a0: QtGui.QFocusEvent):
		self._inFocus = True
		p = self.parentWidget()
		p.installEventFilter(self)
		p.repaint()
		self.focusIn.emit(True)

	def focusOutEvent(self, a0: QtGui.QFocusEvent):
		if not self._isEditing:
			return
		if self._showMenu:
			return
		self.mode = Mode.NONE
		self.focusOut.emit(False)
		self._inFocus = False

	def paintEvent(self, e: QtGui.QPaintEvent):
		pass

	def mousePressEvent(self, e: QtGui.QMouseEvent):
		self.position = QPoint(e.globalX() - self.geometry().x(), e.globalY() - self.geometry().y())
		if not self._isEditing:
			return
		if not self._inFocus:
			return
		if not e.buttons() and QtCore.Qt.LeftButton:
			self.setCursorShape(e.pos())
			return
		if e.button() == QtCore.Qt.RightButton:
			self.popupShow(e.pos())
			e.accept()

	def setCursorShape(self, pos: QPoint):
		diff = 3

		bottom 	= abs(pos.y() - self.y() - self.height()) < diff
		left 	= abs(pos.x() - self.x()) < diff
		right 	= abs(pos.x() - self.x() - self.width()) < diff
		top 	= abs(pos.y() - self.y()) < diff

		if not (bottom or left or right or top):
			self.setCursor(QCursor(QtCore.Qt.SizeAllCursor))
			self.mode = Mode.MOVE
			return

		if bottom and left:
			self.mode = Mode.RESIZEBL
			self.setCursor(QCursor(QtCore.Qt.SizeBDiagCursor))
			return
		elif bottom and right:
			self.mode = Mode.RESIZEBR
			self.setCursor(QCursor(QtCore.Qt.SizeFDiagCursor))
			return
		elif top and left:
			self.mode = Mode.RESIZETL
			self.setCursor(QCursor(QtCore.Qt.SizeFDiagCursor))
			return
		elif top and right:
			self.mode = Mode.RESIZETR
			self.setCursor(QCursor(QtCore.Qt.SizeBDiagCursor))
			return

		if not self._preserveAspectRatio:
			if left:
				self.setCursor(QCursor(QtCore.Qt.SizeHorCursor))
				self.mode = Mode.RESIZEL
			elif right:
				self.setCursor(QCursor(QtCore.Qt.SizeHorCursor))
				self.mode = Mode.RESIZER
			elif top: # Top
				self.setCursor(QCursor(QtCore.Qt.SizeVerCursor))
				self.mode = Mode.RESIZET
			elif bottom: # Bottom
				self.setCursor(QCursor(QtCore.Qt.SizeVerCursor))
				self.mode = Mode.RESIZEB

	def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
		QWidget.mouseReleaseEvent(self, e)

	def mouseMoveEvent(self, e: QtGui.QMouseEvent):
		QWidget.mouseMoveEvent(self, e)
		if not self._isEditing or not self._inFocus:
			return
		if not e.buttons() and QtCore.Qt.LeftButton:
			p = QPoint(e.x() + self.geometry().x(), e.y() + self.geometry().y())
			self.setCursorShape(p)
			return

		#move widget
		if (self.mode == Mode.MOVE or self.mode == Mode.NONE) and e.buttons() and QtCore.Qt.LeftButton:
			toMove = e.globalPos() - self.position
			if toMove.x() < 0: return
			if toMove.y() < 0: return
			if toMove.x() > self.parentWidget().width() - self.width(): return
			self.move(toMove)
			self.newGeometry.emit(self.geometry())
			self.parentWidget().repaint()
			return

		#resize widget
		if (self.mode != Mode.MOVE) and e.buttons() and QtCore.Qt.LeftButton:
			deltaL = e.globalX() - self.position.x() - self.geometry().x()
			deltaT = e.globalY() - self.position.y() - self.geometry().y()
			deltaPos = e.globalPos() - self.position
			if self.mode == Mode.RESIZETL:  # TopLeft
				self.resize(self.geometry().width() - deltaL, self.geometry().height() - deltaT)
				self.move(deltaPos.x(), deltaPos.y())
			elif self.mode == Mode.RESIZETR: # TopRight
				self.resize(e.x(), self.geometry().height() - deltaT)
				self.move(self.x(), deltaPos.y())
			elif self.mode == Mode.RESIZEBL: # BottomLeft
				self.resize(self.geometry().width() - deltaL, e.y())
				self.move(deltaPos.x(), self.y())
			elif self.mode == Mode.RESIZEB:  # Bottom
				self.resize(self.width(), e.y())
			elif self.mode == Mode.RESIZEL:  # Left
				self.resize(self.geometry().width() - deltaL, self.height())
				self.move(deltaPos.x(), self.y())
			elif self.mode == Mode.RESIZET:  # Top
				self.resize(self.width(), self.geometry().height() - deltaT)
				self.move(self.x(), deltaPos.y())
			elif self.mode == Mode.RESIZER:  # Right
				self.resize(e.x(), self.height())
			elif self.mode == Mode.RESIZEBR: # BottomRight
				self.resize(e.x(), e.y())

			self.parentWidget().repaint()
		self.newGeometry.emit(self.geometry())

class GetDimensionsDialog(QDialog):
	""" Dialog window for requesting a width and height input. Can be configured to keep aspect ratio."""
	def __init__(self, h, w, min=1, max=100, constrain=False, parent=None):
		super(GetDimensionsDialog, self).__init__(parent)
		self.setWindowFlags(QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowTitleHint)
		## set default values
		self.h = h
		self.w = w

		self.hL = QLabel("height:")
		self.hInput = QSpinBox(self)
		self.hInput.setMinimum(min)
		self.hInput.setMaximum(max)
		self.hInput.setValue(self.h)

		self.wL = QLabel("width:")
		self.wInput = QSpinBox(self)
		self.wInput.setMinimum(min)
		self.wInput.setMaximum(max)
		self.wInput.setValue(self.w)

		if constrain:
			self.aspectRatio = h/w
			self.hInput.valueChanged.connect(self.constrainW)
			self.wInput.valueChanged.connect(self.constrainH)

		self.OKBtn = QPushButton("OK")
		self.OKBtn.clicked.connect(self.accept)

		## Set layout, add buttons
		layout = QGridLayout()
		layout.setColumnStretch(1, 1)
		layout.setColumnMinimumWidth(1, 250)

		layout.addWidget(self.hL, 0, 0)
		layout.addWidget(self.hInput, 0, 1)
		layout.addWidget(self.wL, 1, 0)
		layout.addWidget(self.wInput, 1, 1)
		layout.addWidget(self.OKBtn, 2, 1)

		self.setLayout(layout)
		self.setWindowTitle("Select grid dimensions")

	def values(self):
		return self.hInput.value(), self.wInput.value()

	def constrainW(self):
		self.wInput.blockSignals(True)
		self.wInput.setValue(self.hInput.value() / self.aspectRatio )
		self.wInput.blockSignals(False)

	def constrainH(self):
		self.hInput.blockSignals(True)
		self.hInput.setValue(self.wInput.value() * self.aspectRatio )
		self.hInput.blockSignals(False)

	@staticmethod
	def getValues(h=1, w=1, min=1, max=100, constrain=False, parent=None):
		dialog = GetDimensionsDialog(h, w, min, max, constrain, parent)
		result = dialog.exec_()
		h, w = dialog.values()
		return h, w, result == QDialog.Accepted

class GetLevelDialog(QDialog):
	""" Dialog window for requesting a level and latent size input."""
	def __init__(self, m, l, parent=None):
		super(GetLevelDialog, self).__init__(parent)
		self.setWindowFlags(QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowTitleHint)
		## Default values

		self.mL = QLabel("merge level:")
		self.mInput = QSpinBox(self)
		self.mInput.setMinimum(2)
		self.mInput.setMaximum(9)
		self.mInput.setValue(m)

		self.lL = QLabel("latent size:")
		self.lInput = QSpinBox(self)
		self.lInput.setMinimum(1)
		self.lInput.setMaximum(512)
		self.lInput.setValue(l)

		self.OKBtn = QPushButton("OK")
		self.OKBtn.clicked.connect(self.accept)

		## Set layout, add buttons
		layout = QGridLayout()
		layout.setColumnStretch(1, 1)
		layout.setColumnMinimumWidth(1, 250)

		layout.addWidget(self.mL, 0, 0)
		layout.addWidget(self.mInput, 0, 1)
		layout.addWidget(self.lL, 1, 0)
		layout.addWidget(self.lInput, 1, 1)
		layout.addWidget(self.OKBtn, 2, 1)

		self.setLayout(layout)
		self.setWindowTitle("Select merge level")

	def values(self):
		return self.mInput.value(), self.lInput.value()

	@staticmethod
	def getValues(m=2, l=1, parent=None):
		dialog = GetLevelDialog(m, l, parent)
		result = dialog.exec_()
		m, l = dialog.values()
		return m, l, result == QDialog.Accepted

class GetDatasetDialog(QDialog):
	""" Dialog window for requesting a dataset selection from a drop-down box."""
	def __init__(self, datasets, dataset, parent=None):
		super(GetDatasetDialog, self).__init__(parent)
		self.setWindowFlags(QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowTitleHint)
		## Default values
		self.dL = QLabel("Choose dataset:")
		self.dInput = QComboBox(self)
		for d in datasets:
			self.dInput.addItem(d)

		self.dInput.setCurrentIndex(self.dInput.findText(dataset))
		self.OKBtn = QPushButton("OK")
		self.OKBtn.clicked.connect(self.accept)

		## Set layout, add buttons
		layout = QGridLayout()
		layout.setColumnStretch(1, 1)
		layout.setColumnMinimumWidth(1, 250)

		layout.addWidget(self.dL, 0, 0)
		layout.addWidget(self.dInput, 0, 1)
		layout.addWidget(self.OKBtn, 1, 1)

		self.setLayout(layout)
		self.setWindowTitle("Select dataset")

	def value(self):
		return self.dInput.currentText()

	@staticmethod
	def getValue(ds=None, d='', parent=None):
		dialog = GetDatasetDialog(ds, d, parent)
		result = dialog.exec_()
		return dialog.value(), result == QDialog.Accepted

class ImageViewer(QtWidgets.QGraphicsView):
	""" Main class for the image viewer widget."""

	imageClicked = Signal(QtCore.QPoint)
	undoCountUpdated = Signal(int)
	guidanceUpdated = Signal(bool)
	imageUpdated = Signal(bool)
	mouseDrop = Signal()
	textEdited = Signal(str)

	def __init__(self, parent):
		super(ImageViewer, self).__init__(parent)
		self.setAcceptDrops(True)
		self._zoom = 0
		self._showGrid = False
		self._empty = True
		self._emptyUnmerged = True
		self._emptyClusters = True
		self.gridSize = QSize(0, 0)
		self.latentSize = 1
		self.mergeLevel = 2
		self.pad = 1
		self.dataset = ''
		self.imageShape = QSize(0, 0)
		self.defaultGuidanceMap = None
		self.guidanceMap = None
		self._scene = QGraphicsScene(self)
		self._image = QGraphicsPixmapItem()
		self._imageClusters = QGraphicsPixmapItem()
		self._imageClusters.setVisible(False)
		self._imageUnmerged = QGraphicsPixmapItem()
		self._imageUnmerged.setVisible(False)
		self._latentIndicator = QGraphicsPixmapItem()
		self._latentIndicator.setVisible(False)
		self._showLatentIndicator = True
		self._scene.addItem(self._image)
		self._scene.addItem(self._imageClusters)
		self._scene.addItem(self._imageUnmerged)
		self._scene.addItem(self._latentIndicator)
		self.setScene(self._scene)
		self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
		self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
		self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
		self.setFrameShape(QtWidgets.QFrame.NoFrame)
		self.latent_dragEvent = False
		self.dragPixmaps = None
		self.drag = None
		self.wheelSelection = 0
		self.latentCluster = None
		self.sampledLatentPos = None
		self.sampledLatent = None
		self.useSimilarLatents = False
		self.mouseClickPosition = None
		self.leftMouseButtonDown = False
		self.middleMouseButtonDown = False
		self._panStart = QPoint(0, 0)

		#show tutorial on startup
		tutorialPath = 'tileGAN_firstSteps.jpg'
		self.updateImage(QtGui.QPixmap(tutorialPath), fitToView=True, updateUI=False)

#-----------------------------------------------------------------------------------------------------------------------------------------
# HANDLING OF INHERITED EVENTS
#-----------------------------------------------------------------------------------------------------------------------------------------
	def dragEnterEvent(self, event):
		event.accept()
		return

	def dragMoveEvent(self, event):
		event.accept()
		return

	def dropEvent(self, event):
		event.accept()
		event.setDropAction(QtCore.Qt.MoveAction)

		self.mouseDrop.emit()
		if self._image.isUnderMouse() and self.latent_dragEvent:
			print('<ImageViewer> latent drop event in ImageViewer')
			modifiers = QtGui.QGuiApplication.keyboardModifiers()

			self.dropLatent(self.latentCluster, event.pos(), modifiers)

		self.latent_dragEvent = False
		self.drag = None
		self.leftMouseButtonDown = False

	def enterEvent(self, event):
		if self.latent_dragEvent:
			return

		if self.hasImage():
			self.updateIndicatorSize()
			if self._showLatentIndicator:
				self._latentIndicator.show()

	def leaveEvent(self, event):
		if self.hasImage():
			self._latentIndicator.hide()

	def mouseMoveEvent(self, event):
		if self.middleMouseButtonDown:
			# panning behaviour
			delta = self._panStart - event.pos()

			self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
			self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
			self._panStart = event.pos()
			event.accept()
			return

		if self.hasImage():
			event.accept()
			self.updateIndicatorPos(event.pos())

		super(ImageViewer, self).mouseMoveEvent(event)

	def mouseReleaseEvent(self, event):
		self.middleMouseButtonDown = False
		self.latent_dragEvent = False

		if self._image.isUnderMouse():
			#processing dragging
			if self.mouseClickPosition is not None and (event.pos() - self.mouseClickPosition).manhattanLength() > QApplication.startDragDistance():
				if not self.leftMouseButtonDown or self.sampledLatent is None:
					super(ImageViewer, self).mousePressEvent(event)
					return

				gridCoords1 = self.getGridCoords(self.mouseClickPosition)
				gridCoords2 = self.getGridCoords(event.pos())

				startX = np.amin([gridCoords1.x(), gridCoords2.x()])
				startY = np.amin([gridCoords1.y(), gridCoords2.y()])
				endX   = np.amax([gridCoords1.x(), gridCoords2.x()])
				endY   = np.amax([gridCoords1.y(), gridCoords2.y()])

				sourceX = self.sampledLatentPos.x()
				sourceY = self.sampledLatentPos.y()

				height = endY - startY
				width =  endX - startX
				if height > 0 and width > 0:
					self.replaceLatentRegion(startX, startY, width, height, sourceX, sourceY)

			#processing click: only do this if not dragging
			elif event.button() & QtCore.Qt.LeftButton:
				print('<ImageViewer> mouseReleaseEvent at {}'.format(self.getPixelCoords(event.pos())))

				modifiers = QtGui.QGuiApplication.keyboardModifiers()

				if self.latentCluster is not None:
					self.dropLatent(self.latentCluster, event.pos(), modifiers)

			self.imageClicked.emit(QtCore.QPoint(event.pos()))

		self.leftMouseButtonDown = False
		super(ImageViewer, self).mousePressEvent(event)

	def mousePressEvent(self, event):
		if self._image.isUnderMouse():
			if event.button() & QtCore.Qt.RightButton:
				print('<ImageViewer> right mouse button clicked!')
				self.sampleLatentAtPos(event.pos())

				modifiers = QtGui.QGuiApplication.keyboardModifiers()
				if modifiers == QtCore.Qt.ShiftModifier:
					self.useSimilarLatents = True

				else:
					self.useSimilarLatents = False

			elif event.button() & QtCore.Qt.MiddleButton:
				print('<ImageViewer> middle mouse button clicked!')
				self.middleMouseButtonDown = True
				self._panStart = event.pos()

			elif event.button() & QtCore.Qt.LeftButton:
				self.leftMouseButtonDown = True
				self.mouseClickPosition = event.pos()
				print('<ImageViewer> left mouse button clicked at {}!'.format(self.mouseClickPosition))

	def wheelEvent(self, event):
		#zoom behaviour
		self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
		if self.hasImage():
			if event.angleDelta().y() > 0:
				factor = 1.25
				self._zoom += 1
			else:
				factor = 0.8
				self._zoom -= 1
			if self._zoom > 0:
				self.scale(factor, factor)
				self.updateIndicatorSize()
			elif self._zoom == 0:
				self.fitInView()
			else:
				self._zoom = 0

	def resizeEvent(self, event):
		self.fitInView()

# -----------------------------------------------------------------------------------------------------------------------------------------

	def setDrag(self, drag):
		self.drag = drag


	def updateIndicatorSize(self, stroke=3, offset=2, crossSize=10):
		"""
		draw a box and crosshair under mouse cursor as rectangle of size latentSize
		"""
		multiplier = 1 #TODO optional: scale indicator with zoom level

		stroke *= multiplier
		offset *= multiplier
		crossSize *= multiplier

		halfStroke = stroke / 2
		rect = self.getImageDims()
		latentSize = self.latentSize * rect.width() / self.gridSize.width()
		halfSize = latentSize / 2
		crossSize = min(crossSize, int(halfSize - 3))

		pixmap = QPixmap(QSize(int(latentSize + stroke + offset), int(latentSize + stroke + offset)))
		#fill rectangle with transparent color
		pixmap.fill(QColor(0,0,0,0)) #transparent

		painter = QPainter(pixmap)
		r = QRectF(QPoint(), QSizeF(latentSize, latentSize))
		r.adjust(offset+halfStroke, offset+halfStroke, -halfStroke, -halfStroke)
		#draw shadow under rectangle
		pen = QPen(QColor(50, 50, 50, 100), stroke) #shadow
		painter.setPen(pen)
		painter.drawRect(r)
		if crossSize > 4:
			painter.drawLine(QPointF(offset+halfSize, offset+halfSize-crossSize), QPointF(offset+halfSize, offset+halfSize+crossSize))
			painter.drawLine(QPointF(offset+halfSize-crossSize, offset+halfSize), QPointF(offset+halfSize+crossSize, offset+halfSize))
		r.adjust(-offset, -offset, -offset, -offset)
		pen = QPen(QColor(styleColor[0], styleColor[1], styleColor[2], 200), stroke)
		painter.setPen(pen)
		painter.drawRect(r)
		if crossSize > 4:
			painter.drawLine(QPointF(halfSize, halfSize - crossSize), QPointF(halfSize, halfSize + crossSize))
			painter.drawLine(QPointF(halfSize - crossSize, halfSize), QPointF(halfSize + crossSize, halfSize))
		painter.end()

		self._latentIndicator.setPixmap(pixmap)

	def updateIndicatorPos(self, eventPos):
		"""
		move indicator cursor to correct position under mouse
		"""
		rect = self.getImageDims()
		gridSize = 	rect.width() / self.gridSize.width()
		latentSize = self.latentSize * gridSize
		centerOffset = latentSize // 2

		scenePos = QGraphicsView.mapToScene(self, eventPos)
		scenePos -= QPointF(centerOffset, centerOffset) #center latentIndicator around mouse

		roundPos = QPoint(int(gridSize * round(scenePos.x()/gridSize)), int(gridSize * round(scenePos.y()/gridSize)))

		roundPos.setX(int(max(-centerOffset, min(roundPos.x(), rect.width()  - ( latentSize - centerOffset )))))
		roundPos.setY(int(max(-centerOffset, min(roundPos.y(), rect.height() - ( latentSize - centerOffset )))))
		self._latentIndicator.setPos(roundPos)

	def hasImage(self):
		return not self._empty

	def getImageDims(self):
		return QRectF(self._image.pixmap().rect())

	def fitInView(self, scale=True):
		rect = self.getImageDims()
		if not rect.isNull():
			self.setSceneRect(rect)

			unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
			self.scale(1 / unity.width(), 1 / unity.height())
			viewrect = self.viewport().rect()
			scenerect = self.transform().mapRect(rect)
			factor = min(viewrect.width() / scenerect.width(),
						 viewrect.height() / scenerect.height())
			self.scale(factor, factor)

			if self.hasImage():
				self.updateIndicatorSize()

			self._zoom = 0

	def toggleGrid(self):
		"""
		toggle the foreground grid that indicates the latent grid spacing
		"""
		if not self.hasImage():
			return
		self._showGrid = not self._showGrid
		self._scene.update()

	def toggleLatentIndicator(self, state):
		"""
		toggle the crosshair that indicates the latent size
		"""
		self._showLatentIndicator = state

	def dragStarted(self, latentCluster):
		"""
		start a drag operation in imageViewer
		"""
		print('<ImageViewer> drag started')
		self.latent_dragEvent = True
		self.latentCluster = latentCluster

	def activateLatentCluster(self, latentCluster):
		self.latentCluster = latentCluster

	def getPixelCoords(self, pos):
		floatPos = QGraphicsView.mapToScene(self, pos)
		return QPoint(np.rint(floatPos.x()), np.rint(floatPos.y()))

	def roundToGlobalGridCoords(self, pos):
		"""
		return a global coordinate for a latent grid position
		"""
		roundedGridPos = self.getGridCoords(pos)
		imageRect = self.getImageDims()
		pixelPos = QPointF(roundedGridPos.x() * imageRect.width() / self.gridSize.width(), roundedGridPos.y() * imageRect.height() / self.gridSize.height())
		return QGraphicsView.mapToGlobal(self, QGraphicsView.mapFromScene(self, pixelPos))

	def getGridCoords(self, pos):
		"""
		return a grid coordinate for a global position
		"""
		imageRect = self.getImageDims()
		pixelCoords = self.getPixelCoords(pos)
		gridX = np.rint(self.gridSize.width()  * pixelCoords.x() / imageRect.width())
		gridY = np.rint(self.gridSize.height() * pixelCoords.y() / imageRect.height())

		return QPoint(gridX, gridY)

	def getCurrentLatentSize(self):
		"""
		return the size in pixels of a latent at current zoom level
		"""
		rect = self.getImageDims()
		scenerect = self.transform().mapRect(rect)
		latentSize = self.latentSize * scenerect.width() / self.gridSize.width() #(self.latentSize  *
		return latentSize

	def dropLatent(self, latentCluster, pos, keyModifiers):
		gridCoords = self.getGridCoords(pos)
		alpha = 0.1
		if keyModifiers == QtCore.Qt.ShiftModifier:
			print('<ImageViewer> perturbing latent at ({}, {}) with class {}'.format(gridCoords.x(), gridCoords.y(), latentCluster))
			sourceX = self.sampledLatentPos.x()
			sourceY = self.sampledLatentPos.y()
			output, undoCount = np.asarray(tf_manager.perturbLatent(gridCoords.x(), gridCoords.y(), sourceX, sourceY, alpha)._getvalue())
		else:
			print('<ImageViewer> dropped latent of class {} at ({}, {})'.format(latentCluster, gridCoords.x(), gridCoords.y()))
			output, undoCount = np.asarray(tf_manager.putLatent(gridCoords.x(), gridCoords.y(), latentCluster)._getvalue())
		self.undoCountUpdated.emit(undoCount)
		self.updateImage(output)

	def replaceLatentRegion(self, startX, startY, width, height, sampleX, sampleY):#, x_target, y_target):
		mode = 'similar' if self.useSimilarLatents else 'identical'
		#mode = 'cluster' #if self.useSimilarLatents else 'identical'
		print('<ImageViewer> replaceLatentRegion at ({}, {}) of size {}x{} from ({}, {}) with {}'.format(startX, startY, width, height, sampleX, sampleY, mode))
		output, undoCount = np.asarray(tf_manager.pasteLatents(self.sampledLatent, startX, startY, width, height, sampleX, sampleY, mode)._getvalue())
		self.undoCountUpdated.emit(undoCount)
		self.updateImage(output)

	def undo(self):
		output, undoCount = np.asarray(tf_manager.undo()._getvalue())
		print('<ImageViewer> undo (new undo count = {})'.format(undoCount))
		self.undoCountUpdated.emit(undoCount)
		self.updateImage(output)

	def toggleMerging(self, showMerging):
		"""
		toggle view between merged latents and unmerged grid
		"""
		if not self.hasImage():
			return
		if not showMerging and self._emptyUnmerged:
			output = np.asarray(tf_manager.getUnmergedOutput()._getvalue())
			pixmap = self.pixmapFromArray(output)
			self._imageUnmerged.setPixmap(pixmap)
			self._emptyUnmerged = False
		self._imageUnmerged.setVisible(not showMerging)

	def toggleClusters(self, showClusters):
		"""
		toggle view between merged latents and clusters of each latent
		"""
		if not self.hasImage():
			return
		if showClusters and self._emptyClusters:
			print('<ImageViewer> toggle clusters')
			output = np.asarray(tf_manager.getClusterOutput()._getvalue())
			rect = self.getImageDims()
			upsampledOutput = output.repeat(rect.height()//output.shape[0], axis=0).repeat(rect.width()//output.shape[1], axis=1)
			pixmap = self.pixmapFromArray(upsampledOutput)
			self._imageClusters.setPixmap(pixmap)
			self._emptyClusters = False
		self._imageClusters.setVisible(showClusters)

	def toggleGuidanceMap(self):
		"""
		hide or show floating guidance viewer widget as a layover on top of image viewer
		"""
		self.guidanceViewer.toggle()

	def refresh(self):
		"""
		force an update of the texture from manager
		"""
		print('<ImageViewer> refresh')
		output, gridShape = np.asarray(tf_manager.getOutput()._getvalue())
		self.updateGridShape(np.asarray(gridShape))
		self.updateImage(output)

	def improveResults(self):
		print('<ImageViewer> improve')
		output = np.asarray(tf_manager.improveLatents()._getvalue())
		self.updateImage(output)

	def randomize(self):
		"""
		generate a randomized latent grid
		"""
		self.defaultGuidanceMap = None
		self.guidanceMap = None
		self.guidanceUpdated.emit(False)

		w = self.gridSize.width() // self.latentSize
		h = self.gridSize.height() // self.latentSize

		if w < 1 and h < 1:
			w = 5
			h = 3
		h, w, ok = GetDimensionsDialog.getValues(h=max(h, 1), w=max(w, 1))

		if not ok: #Dialog was terminated
			return
		print('<ImageViewer> randomizing {}x{} grid'.format(h, w))

		output, gridShape, undoCount = tf_manager.randomizeGrid(h, w)._getvalue()
		self.undoCountUpdated.emit(undoCount)
		self.updateGridShape(np.asarray(gridShape))
		self.updateImage(np.asarray(output), fitToView=True)


	def deadLeaves(self):
		"""
		generate a randomized latent grid with larger coherent regions using "dead leaves" algorithm
		"""
		self.defaultGuidanceMap = None
		self.guidanceMap = None
		self.guidanceUpdated.emit(False)

		w = self.gridSize.width() // self.latentSize
		h = self.gridSize.height() // self.latentSize

		h, w, ok = GetDimensionsDialog.getValues(h=max(h, 1), w=max(w, 1))

		if not ok: #Dialog was terminated
			return
		print('randomizing {}x{} grid'.format(h, w))

		output, gridShape, undoCount = tf_manager.deadLeaves(h, w)._getvalue()

		self.undoCountUpdated.emit(undoCount)
		self.updateGridShape(np.asarray(gridShape))
		self.updateImage(np.asarray(output), fitToView=True)

	def setMergeLevel(self):
		"""
		adjust latent merge level
		"""
		level, latentSize, ok = GetLevelDialog.getValues(m=self.mergeLevel, l=self.latentSize)

		#check parameter validity
		if latentSize > 2 ** level:
			latentSize = 2 ** level
			msg = QMessageBox()
			msg.setWindowTitle("Latent settings warning")
			msg.setIcon(QMessageBox.Warning)
			msg.setText("Latent size cannot be larger than 2^level.\n Latent size for level {} reset to {}".format(level, latentSize))
			msg.setStandardButtons(QMessageBox.Ok)
			msg.exec_()

		if not ok or (level == self.mergeLevel and latentSize == self.latentSize): #Dialog was terminated or settings stayed the same
			return

		self.mergeLevel = level
		self.latentSize = latentSize
		print('<ImageViewer> updating new merge level to {}, latentSize {}'.format(self.mergeLevel, self.latentSize))
		tf_manager.setMergeLevel(self.mergeLevel, self.latentSize)
		output, gridShape = np.asarray(tf_manager.getOutput()._getvalue())
		self.updateGridShape(np.asarray(gridShape))
		self.updateImage(output, fitToView=True)

	def resizeGuidanceMap(self, noUpdate=False):
		"""
		rescale the guidance image to generate a different result size. guidance image is stored in original resolution and always resized from original image.
		"""
		guidanceImg = Image.fromarray(self.defaultGuidanceMap)
		if noUpdate:
			w = guidanceImg.width
			h = guidanceImg.height
		else:
			if self.guidanceMap is not None:
				h = self.guidanceMap.shape[0]
				w = self.guidanceMap.shape[1]
			else:
				h = self.defaultGuidanceMap.shape[0]
				w = self.defaultGuidanceMap.shape[1]

		h, w, ok = GetDimensionsDialog.getValues(h=h, w=w, max=10000, constrain=True)

		print('h {} w {} ok {}'.format(h, w, ok))
		if not ok: #Dialog was terminated
			return False

		self.guidanceMap = np.asarray(guidanceImg.resize((w, h)))
		if not noUpdate:
			print('updating now!')
			self.update()
		else:
			self.guidanceUpdated.emit(True)
		return True

	def update(self):
		output, descriptorGrid, gridShape, undoCount = np.asarray(tf_manager.getUpsampled(self.guidanceMap)._getvalue())
		self.updateImage(output, fitToView=True)
		self.undoCountUpdated.emit(undoCount)
		self.updateGridShape(gridShape)

	def processImage(self, fname):
		"""
		load image from file, convert image, request desired guidance map size and set as guidance image
		"""
		pilImg = Image.open(fname)

		self.defaultGuidanceMap = np.asarray(pilImg.convert("RGB"))

		success = self.resizeGuidanceMap(noUpdate=True)
		if not success: #Dialog was terminated
			return

		# TODO don't allow arbitrarily big images
		self.guidanceViewer.setImage(self.pixmapFromArray(self.defaultGuidanceMap))

		output, descriptorGrid, gridShape, undoCount = np.asarray(tf_manager.getUpsampled(self.guidanceMap)._getvalue())
		self.undoCountUpdated.emit(undoCount)
		self.updateGridShape(gridShape)
		self.updateImage(output, fitToView=True)

	def updateGridShape(self, gridShape):
		self.gridSize = QSize(gridShape[1], gridShape[0])
		self.latentSize = gridShape[2]
		self.mergeLevel = gridShape[3]

	def drawForeground(self, painter, rect):
		"""
		override drawForeground method to paint latent grid cells
		"""
		if self._showGrid:
			rect = self.getImageDims()
			#draw all grid cells
			penNarrow = QPen(QColor(100, 100, 100, 150), 2)
			penNarrow.setStyle(QtCore.Qt.CustomDashLine)
			penNarrow.setDashPattern([1, 2])

			penBold = QPen(QColor(styleColor[0], styleColor[1], styleColor[2], 150), 3)
			penBold.setStyle(QtCore.Qt.CustomDashLine)
			penBold.setDashPattern([1, 2])

			painter.setPen(penNarrow)

			d = rect.width() / self.gridSize.width()
			for y in range(1, self.gridSize.height()):
				if y % self.latentSize == 0: #bold lines at more important grid positions
					painter.setPen(penBold)
				else:
					painter.setPen(penNarrow)
				painter.drawLine(QPoint(0, int(y * d)), QPoint(int(rect.width()), int(y * d)))

			for x in range(1, self.gridSize.width()):
				if x % self.latentSize == 0:
					painter.setPen(penBold)
				else:
					painter.setPen(penNarrow)
				painter.drawLine(QPoint(int(x*d), 0), QPoint(int(x*d), int(rect.height())))

	def pixmapFromArray(self, array):
		"""
		convert numpy array to QPixmap. Memory needs to be copied to avoid mem issues.
		"""
		print(array.shape)
		if array.shape[0] == 3:
			array = np.rollaxis(array, 0, 3)

		self.imageShape = QSize(array.shape[1], array.shape[0])
		cp = array.copy()
		image = QImage(cp, array.shape[1], array.shape[0], QImage.Format_RGB888)
		return QPixmap(image)

	def updateImage(self, image=None, fitToView=False, updateUI=True):
		"""
		update result image in image viewer and trigger all appropriate settings
		"""
		if image is not None:
			if type(image) is np.ndarray: #if input type is numpy array, convert to pixmap
				image = self.pixmapFromArray(image)

			self._image.setPixmap(image)

			if fitToView:
				self.fitInView()

			if updateUI:
				self.textEdited.emit('merge level: %d | latent size: %d | image size: %dx%d px | %.2f MP' % (self.mergeLevel, self.latentSize, image.width(), image.height(), float(image.width() * image.height()) / 1000000))
				self._empty = False

		else:
			self._empty = True

			self._image.setPixmap(QtGui.QPixmap())
			self._latentIndicator.hide()
			self._showLatentIndicator = False
		self._emptyUnmerged = True
		self._emptyClusters = True

		self.imageUpdated.emit(not self._empty)
		#update unmerged image, if necessary
		if self._imageUnmerged.isVisible():
			self.toggleMerging(False)

	def saveLatents(self):
		"""
		save latents to file. currently pointless, as loading from file is not implemented.
		"""
		tf_manager.saveLatents()

	def saveImage(self):
		filename = QFileDialog.getSaveFileName(self, 'Save image as...', str(Path.home())+'\Desktop')

		if filename is None or filename == "":
			return

		print(filename)
		self._image.pixmap().save(filename[0], "JPG")

	def sampleLatentAtPos(self, pos):
		imgRect = self.getImageDims()
		gridDims = self.gridSize #self.latentSize *
		w = int(imgRect.width() / self.gridSize.width()) #self.latentSize * ( #self.getCurrentLatentWidth()
		h_l = self.latentSize // 2

		self.sampledLatentPos = self.getGridCoords(pos)

		posGrid = self.sampledLatentPos - QPoint(h_l, h_l)
		posGrid.setX(min(max(posGrid.x(), 0), gridDims.width() - h_l))
		posGrid.setY(min(max(posGrid.y(), 0), gridDims.height() - h_l))

		posImg = QPoint(int(imgRect.width() * posGrid.x() / gridDims.width()), int(imgRect.height() * posGrid.y() / gridDims.height()))

		fullImg = self._image.pixmap()
		imgRegion = fullImg.copy(QRect(posImg, QSize(w, w))).toImage()

		region_array = np.ndarray(shape=(imgRegion.height(), imgRegion.width(), 4), dtype=np.uint8, buffer=imgRegion.bits())
		self.sampledLatent = region_array[:, :, [2, 1, 0]] #BGR -> RGB

	def toggleDragMode(self):
		if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
			self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
		elif not self._image.pixmap().isNull():
			self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

class LatentLabel(QLabel):
	latentDragged = Signal(int)
	activated = Signal(int)

	def __init__(self, parent, index, imageArray, customColor=None):
		super(LatentLabel, self).__init__(parent)
		self.mainWidget = parent
		self.latentCluster = index
		cp = imageArray.copy()
		image = QImage(cp, imageArray.shape[1], imageArray.shape[0], QImage.Format_RGB888)

		self.image = QPixmap(image)
		self.setPixmap(self.image.scaled(128, 128))
		self.setGeometry(0, 0, 128, 128)
		self.dragStartPosition = QPoint()
		self.isActive = False
		self.setStyleSheet('border: none')
		self.customColor = customColor #RGB color tuple
		self.mouseDown = False

	def setActive(self):
		self.isActive = True
		if self.customColor is None:
			self.setStyleSheet("border: 5px solid #%02x%02x%02x" % styleColor)
		else:
			self.setStyleSheet("border: 5px solid #%02x%02x%02x" % self.customColor)
		self.activated.emit(self.latentCluster)

	def setInactive(self):
		self.isActive = False
		self.setStyleSheet('border: none')

	def toggleActive(self):
		if self.isActive:
			self.setInactive()
		else:
			self.setActive()

	def dropFinished(self):
		self.mouseDown = False
		self.setInactive()

	def getName(self):
		return 'cluster_{}'.format(self.latentCluster)

	def mouseMoveEvent(self, event):
		if (event.pos() - self.dragStartPosition).manhattanLength() < QApplication.startDragDistance():
			return

		event.accept()
		print('starting drag event')
		self.latentDragged.emit(self.latentCluster)

		self.setActive()

		width = self.mainWidget.getCurrentLatentSize()

		dragPix = self.image.scaled(width, width)
		drag = QDrag(self)
		drag.setMimeData(QMimeData(self))
		drag.setPixmap(dragPix)
		drag.setHotSpot(QPoint(drag.pixmap().width() // 2, drag.pixmap().height() // 2))
		self.mainWidget.viewer.setDrag(drag)
		dropAction = drag.start(QtCore.Qt.MoveAction)

		print('created drag with pixmap of size {}x{}'.format(width, width))

	def mousePressEvent(self, event):
		event.accept()
		self.mouseDown = True
		self.dragStartPosition = event.pos()
		print('mouse pressed in', self.latentCluster)

	def mouseReleaseEvent(self, event):
		event.accept()
		if self.mouseDown:
			self.toggleActive()

class MainWidget(QtWidgets.QWidget):
	def __init__(self):
		super(MainWidget, self).__init__()
		self.viewer = ImageViewer(self)
		self.guidanceViewer = None

		self.viewer.textEdited.connect(self.setInformation)

		self.btnDataset = QToolButton(self)
		self.btnDataset.setIcon(QtGui.QIcon(iconFolder + '/icon_dataset.png'))
		self.btnDataset.setToolTip('Select Dataset')
		self.btnDataset.setIconSize(QSize(32, 32))
		self.btnDataset.clicked.connect(self.setDataset)
		self.dataset = ''

		# 'Load image' button
		self.btnLoad = QToolButton(self)
		self.btnLoad.setIcon(QtGui.QIcon(iconFolder + '/icon_upload.png'))
		self.btnLoad.setToolTip('Open Image')
		self.btnLoad.setIconSize(QSize(32, 32))
		self.btnLoad.clicked.connect(self.loadImageDialog)

		self.btnSave = QToolButton(self)
		icon = QIcon()
		icon.addPixmap(QPixmap(iconFolder + '/icon_download.png'), QIcon.Normal)
		icon.addPixmap(QPixmap(iconFolder + '/icon_download_disabled.png'), QIcon.Disabled)
		self.btnSave.setIcon(icon)
		self.btnSave.setToolTip('Save Image')
		self.btnSave.setEnabled(False)
		self.btnSave.setIconSize(QSize(32, 32))
		self.btnSave.clicked.connect(self.viewer.saveImage)
		self.viewer.imageUpdated.connect(self.setImageEnabled)

		self.btnSaveLatents = QToolButton(self)
		self.btnSaveLatents.setIcon(QtGui.QIcon(iconFolder + '/icon_download_latents.png'))
		self.btnSaveLatents.setToolTip('Save Latents')
		self.btnSaveLatents.setIconSize(QSize(32, 32))
		self.btnSaveLatents.setEnabled(False)
		self.btnSaveLatents.clicked.connect(self.viewer.saveLatents)

		self.btnUndo = QToolButton(self)
		icon = QIcon()
		icon.addPixmap(QPixmap(iconFolder + '/icon_undo.png'), QIcon.Normal)
		icon.addPixmap(QPixmap(iconFolder + '/icon_undo_disabled.png'), QIcon.Disabled)
		self.btnUndo.setIcon(icon)
		self.btnUndo.setToolTip('Undo')
		self.btnUndo.setEnabled(False)
		self.btnUndo.setIconSize(QSize(32, 32))
		self.btnUndo.clicked.connect(self.viewer.undo)
		self.viewer.undoCountUpdated.connect(self.setUndoEnabled)

		self.btnRandomize = QToolButton(self)
		self.btnRandomize.setIcon(QtGui.QIcon(iconFolder + '/icon_randomize.png'))
		self.btnRandomize.setIconSize(QSize(32, 32))
		self.btnRandomize.setToolTip('Randomize')
		self.btnRandomize.clicked.connect(self.viewer.randomize)

		self.btnDeadLeaves = QToolButton(self)
		self.btnDeadLeaves.setIcon(QtGui.QIcon(iconFolder + '/icon_randomize_regions.png'))
		self.btnDeadLeaves.setIconSize(QSize(32, 32))
		self.btnDeadLeaves.setToolTip('Randomize larger regions')
		self.btnDeadLeaves.clicked.connect(self.viewer.deadLeaves)

		self.btnRefresh = QToolButton(self)
		self.btnRefresh.setIcon(QtGui.QIcon(iconFolder + '/icon_refresh.png'))
		self.btnRefresh.setIconSize(QSize(32, 32))
		self.btnRefresh.setToolTip('Refresh')
		self.btnRefresh.clicked.connect(self.viewer.refresh)

		self.btnSetLvl = QToolButton(self)
		self.btnSetLvl.setIcon(QtGui.QIcon(iconFolder + '/icon_merge_level.png'))
		self.btnSetLvl.setIconSize(QSize(32, 32))
		self.btnSetLvl.setToolTip('Set Merge Level')
		self.btnSetLvl.clicked.connect(self.viewer.setMergeLevel)

		self.btnMerged = QToolButton(self)
		self.btnMerged.setCheckable(True)
		self.btnMerged.setEnabled(False)
		self.btnMerged.setChecked(True)
		icon = QIcon()
		icon.addPixmap(QPixmap(iconFolder + '/icon_merged_disabled.png'), QIcon.Disabled)
		icon.addPixmap(QPixmap(iconFolder + '/icon_merged.png'), QIcon.Normal, QIcon.On)
		icon.addPixmap(QPixmap(iconFolder + '/icon_unmerged.png'), QIcon.Normal, QIcon.Off)
		self.btnMerged.setIcon(icon)
		self.btnMerged.setToolTip('Toggle merging')
		self.btnMerged.setIconSize(QSize(32, 32))
		self.btnMerged.toggled.connect(self.viewer.toggleMerging)

		self.btnClusters = QToolButton(self)
		self.btnClusters.setCheckable(True)
		self.btnClusters.setEnabled(False)
		self.btnClusters.setChecked(False)
		icon = QIcon()
		icon.addPixmap(QPixmap(iconFolder + '/icon_clusters_disabled.png'), QIcon.Disabled)
		icon.addPixmap(QPixmap(iconFolder + '/icon_clusters.png'), QIcon.Normal, QIcon.On)
		icon.addPixmap(QPixmap(iconFolder + '/icon_no_clusters.png'), QIcon.Normal, QIcon.Off)
		self.btnClusters.setIcon(icon)
		self.btnClusters.setToolTip('Toggle clusters')
		self.btnClusters.setIconSize(QSize(32, 32))
		self.btnClusters.toggled.connect(self.viewer.toggleClusters)

		self.btnImprove = QToolButton(self)
		self.btnImprove.setIcon(QtGui.QIcon(iconFolder + '/icon_fix.png'))
		self.btnImprove.setIconSize(QSize(32, 32))
		self.btnImprove.setToolTip('Improve')
		self.btnImprove.clicked.connect(self.viewer.improveResults)

		self.btnResize = QToolButton(self)
		icon = QIcon()
		icon.addPixmap(QPixmap(iconFolder + '/icon_image_resize_disabled.png'), QIcon.Disabled)
		icon.addPixmap(QPixmap(iconFolder + '/icon_image_resize.png'))
		self.btnResize.setIcon(icon)
		self.btnResize.setIconSize(QSize(32, 32))
		self.btnResize.setEnabled(False)
		self.btnResize.setToolTip('Resize Guidance Map')
		self.btnResize.clicked.connect(self.viewer.resizeGuidanceMap)

		self.btnGuidance = QToolButton(self)
		self.btnGuidance.setCheckable(True)
		icon = QIcon()
		icon.addPixmap(QPixmap(iconFolder + '/icon_image_disabled.png'), QIcon.Disabled)
		icon.addPixmap(QPixmap(iconFolder + '/icon_image.png'), QIcon.Normal, QIcon.On)
		icon.addPixmap(QPixmap(iconFolder + '/icon_image_off.png'), QIcon.Normal, QIcon.Off)
		self.btnGuidance.setIcon(icon)
		self.btnGuidance.setIconSize(QSize(32, 32))
		self.btnGuidance.setChecked(True)
		self.btnGuidance.setEnabled(False)
		self.btnGuidance.setToolTip('Show Guidance Map')
		self.btnGuidance.toggled.connect(self.viewer.toggleGuidanceMap)

		self.viewer.guidanceUpdated.connect(self.setGuidanceEnabled)

		self.infoTextBox = QLineEdit(self)
		self.infoTextBox.setReadOnly(True)
		self.viewer.imageClicked.connect(self.imageClicked)

		self.latentPicker = QWidget()
		self.LatentPickerLayout = QHBoxLayout(self.latentPicker)
		self.LatentPickerLayout.setContentsMargins(0, 0, 0, 0)
		self.LatentPickerLayout.setAlignment(QtCore.Qt.AlignCenter)
		self.createLatentPicker(self.LatentPickerLayout)
		self.latentPicker.setLayout(self.LatentPickerLayout)

		self.btnLatents = QToolButton(self)
		self.btnLatents.setCheckable(True)
		icon = QIcon()
		icon.addPixmap(QPixmap(iconFolder + '/icon_latents_disabled.png'), QIcon.Normal, QIcon.Off)
		icon.addPixmap(QPixmap(iconFolder + '/icon_latents.png'), QIcon.Normal, QIcon.On)
		self.btnLatents.setIcon(icon)
		self.btnLatents.setIconSize(QSize(32, 32))
		self.btnLatents.setToolTip('Show latent picker')
		self.btnLatents.setChecked(True)
		self.btnLatents.toggled.connect(self.toggleShowLatentPicker)

		self.btnIndicator = QToolButton(self)
		self.btnIndicator.setCheckable(True)
		icon = QIcon()
		icon.addPixmap(QPixmap(iconFolder + '/icon_crosshair_off.png'), QIcon.Normal, QIcon.Off)
		icon.addPixmap(QPixmap(iconFolder + '/icon_crosshair.png'), QIcon.Normal, QIcon.On)
		self.btnIndicator.setIcon(icon)
		self.btnIndicator.setIconSize(QSize(32, 32))
		self.btnIndicator.setToolTip('Show latent indicator')
		self.btnIndicator.setChecked(True)
		self.btnIndicator.toggled.connect(self.viewer.toggleLatentIndicator)

		self.btnGrid = QToolButton(self)
		self.btnGrid.setCheckable(True)
		icon = QIcon()
		icon.addPixmap(QPixmap(iconFolder + '/icon_grid_disabled.png'), QIcon.Disabled)
		icon.addPixmap(QPixmap(iconFolder + '/icon_grid_off.png'), QIcon.Normal, QIcon.Off)
		icon.addPixmap(QPixmap(iconFolder + '/icon_grid.png'), QIcon.Normal, QIcon.On)
		self.btnGrid.setIcon(icon)
		self.btnGrid.setIconSize(QSize(32, 32))
		self.btnGrid.setToolTip('Show grid')
		self.btnGrid.setChecked(False)
		self.btnGrid.setEnabled(False)
		self.btnGrid.toggled.connect(self.viewer.toggleGrid)

		# Arrange layout
		VBlayout = QVBoxLayout(self)

		self.viewer.guidanceViewer = FloatViewer(self, QPoint(15, 15), preserveAspectRatio=True)
		self.viewer.guidanceViewer.hide()

		VBlayout.addWidget(self.viewer)
		VBlayout.addWidget(self.latentPicker)

		HBlayout = QHBoxLayout()
		HBlayout.setAlignment(QtCore.Qt.AlignLeft)
		HBlayout.addWidget(self.btnDataset)

		HBlayout.addWidget(self.btnUndo)
		HBlayout.addWidget(self.btnRandomize)
		HBlayout.addWidget(self.btnDeadLeaves)
		HBlayout.addWidget(self.btnImprove)
		HBlayout.addWidget(self.btnSetLvl)
		HBlayout.addWidget(self.btnRefresh)
		HBlayout.addWidget(self.btnLoad)
		HBlayout.addWidget(self.btnSave)
		HBlayout.addWidget(self.btnGuidance)
		HBlayout.addWidget(self.btnResize)
		HBlayout.addWidget(self.btnSaveLatents)
		HBlayout.addWidget(self.infoTextBox)
		HBlayout.addWidget(self.btnClusters)
		HBlayout.addWidget(self.btnMerged)
		HBlayout.addWidget(self.btnGrid)
		HBlayout.addWidget(self.btnIndicator)
		HBlayout.addWidget(self.btnLatents)

		VBlayout.addLayout(HBlayout)
		self.viewer.fitInView()

	def createLatentPicker(self, widgetLayout):
		def clearLayout(layout):
			while layout.count():
				child = layout.takeAt(0)
				if child.widget():
					child.widget().deleteLater()

		clearLayout(widgetLayout)

		latent_images = np.asarray(tf_manager.getLatentImages()._getvalue())

		latent_colors = np.asarray(tf_manager.getDominantClusterColors()._getvalue())
		latent_hues = [ colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)[0] for r, g, b in latent_colors ]
		sorting = np.argsort(latent_hues)

		# create latent picker label widgets
		self.latentLabels = []
		for i in range(len(latent_images)):
			cluster = sorting[i]
			label = LatentLabel(self, cluster, latent_images[cluster], tuple(latent_colors[cluster]))
			label.latentDragged.connect(self.viewer.dragStarted)
			label.activated.connect(self.activateLabel)
			self.latentLabels.append(label)
			self.viewer.mouseDrop.connect(label.dropFinished)
			widgetLayout.addWidget(label)

	def setDataset(self):
		#request available datasets from server
		datasets, currentDataset = np.asarray(tf_manager.findDatasets()._getvalue())
		dataset, ok = GetDatasetDialog.getValue(datasets, currentDataset)

		if not ok or dataset == currentDataset:
			return

		self.dataset = dataset
		tf_manager.initDataset(self.dataset)
		self.createLatentPicker(self.LatentPickerLayout)

	def setUndoEnabled(self, undoCount):
		print('called setUndoEnabled with count', undoCount)
		self.btnUndo.setEnabled(undoCount > 0)

	def setImageEnabled(self, state):
		self.btnSave.setEnabled(state)
		self.btnMerged.setEnabled(state)
		self.btnClusters.setEnabled(state)
		self.btnGrid.setEnabled(state)

	def setGuidanceEnabled(self, state):
		self.btnResize.setEnabled(state)
		self.btnGuidance.setEnabled(state)
		if not state and self.guidanceViewer is not None:
			self.guidanceViewer.hide()

	def activateLabel(self, activeLabel=-1):
		if activeLabel != -1:
			self.viewer.activateLatentCluster(activeLabel)

		for label in self.latentLabels:
			#deactivate all other labels
			if label.latentCluster != activeLabel:
				label.setInactive()

	def getCurrentLatentSize(self):
		return self.viewer.getCurrentLatentSize()

	def dragEnterEvent(self, event):
		print('parent gets dragEnterEvent')

	def mouseReleaseEvent(self, event):
		print('parent gets mouseReleaseEvent')
		self.viewer.mouseReleaseEvent(event)

	def loadImageDialog(self):
		#Get the file locationR
		filename, _ = QFileDialog.getOpenFileName(self, 'Open file', str(Path.home())+'\Desktop')

		if filename is None or filename == "":
			return

		self.fname = filename
		# Load the image from the location
		self.viewer.processImage(self.fname)

	def toggleShowLatentPicker(self, state):
		self.latentPicker.setVisible(state)
		self.viewer.fitInView()

	def pixInfo(self):
		self.viewer.toggleDragMode()

	def setInformation(self, str):
		self.infoTextBox.setText(str)

	def imageClicked(self, pos):
		return

class MainWindow(QMainWindow):
	resized = Signal()

	def __init__(self, widget):
		QMainWindow.__init__(self)
		#self.setWindowFlags(QtCore.Qt.CustomizeWindowHint)
		self.setWindowTitle("TileGAN")
		app_icon = QtGui.QIcon()
		app_icon.addFile(iconFolder + '/icon_tilegan_16x16.png', QtCore.QSize(16, 16))
		app_icon.addFile(iconFolder + '/icon_tilegan_24x24.png', QtCore.QSize(24, 24))
		app_icon.addFile(iconFolder + '/icon_tilegan_32x32.png', QtCore.QSize(32, 32))
		app_icon.addFile(iconFolder + '/icon_tilegan_48x48.png', QtCore.QSize(48, 48))
		app_icon.addFile(iconFolder + '/icon_tilegan_64x64.png', QtCore.QSize(64, 64))
		self.setWindowIcon(app_icon)

		## Exit Action
		exit_action = QAction("Exit", self)
		exit_action.setShortcut(QtGui.QKeySequence("Ctrl+Q"))#)
		exit_action.triggered.connect(self.exit_app)

		# Window dimensions
		self.setCentralWidget(widget)

		geometry = app.desktop().availableGeometry(self)
		self.resize(int(geometry.height() * 0.85), int(geometry.height() * 0.75))

	@Slot()
	def exit_app(self, checked):
		sys.exit()

class Server(BaseManager): pass

def getServer(ip='', port=8080):
	print('Attempting to connect to TensorFlow manager at {}:{}...'.format(ip, port))
	server = Server(address=(ip, port), authkey=b'tilegan') #localhost
	server.register('sampleFromCluster')
	server.register('findDatasets')
	server.register('initDataset')
	server.register('getLatentImages')
	server.register('getLatentAverages')
	server.register('getDominantClusterColors')
	server.register('getUpsampled')
	server.register('putLatent')
	server.register('perturbLatent')
	server.register('getOutput')
	server.register('getUnmergedOutput')
	server.register('getClusterOutput')
	server.register('getClusterAt')
	server.register('pasteLatents')
	server.register('saveLatents')
	server.register('improveLatents')
	server.register('setMergeLevel')
	server.register('undo')
	server.register('randomizeGrid')
	server.register('deadLeaves')
	connected = False
	attempts = 0
	max_attempts = 10
	while not connected and attempts < max_attempts:
		try:
			server.connect()
			connected = True
			print('Connected to server!')
		except ConnectionRefusedError:
			attempts += 1
			print('Server not ready, waiting ({}/{})...'.format(attempts, max_attempts))
			time.sleep(3)
			pass
	return server, connected

if __name__ == '__main__':
	import argparse
	import sys

	parser = argparse.ArgumentParser(description='Starts the TileGAN user interface.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('address', nargs='?', metavar='ip', type=str, default='localhost', help='IP address of Tensorflow manager.')
	parser.add_argument('port', nargs='?', metavar='port', type=int, default=8080, help='Specify the port for communicating to Tensorflow manager.')

	args = parser.parse_args()

	tf_manager, connected = getServer(ip=args.address, port=args.port)

	if not connected:
		print('Connection could not be established, quitting.')
		exit()

	app = QtWidgets.QApplication(sys.argv)

	if USE_DARK_THEME:
		app.setStyleSheet(qdarkstyle.load_stylesheet())

	widget = MainWidget()

	# QMainWindow using QWidget as central widget
	window = MainWindow(widget)
	window.show()
	sys.exit(app.exec_())