import sys
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, 
    QGroupBox, QCheckBox, QSizePolicy, QStatusBar, QGraphicsView, 
    QGraphicsScene, QGraphicsPixmapItem
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QMouseEvent
from PySide6.QtCore import Qt, QEvent, QPoint

STYLE_SHEET = """
QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 14px;
    padding-top: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
}
QPushButton {
    background-color: #0e639c;
    border: none;
    color: white;
    padding: 6px 12px;
    border-radius: 3px;
}
QPushButton:hover { background-color: #1177bb; }
QPushButton:pressed { background-color: #094771; }
QComboBox, QDoubleSpinBox {
    background-color: #2d2d2d;
    border: 1px solid #3c3c3c;
    padding: 4px;
    border-radius: 3px;
}
QCheckBox { spacing: 8px; }
QStatusBar { background-color: #007acc; color: white; }
QGraphicsView {
    border: 1px solid #3c3c3c;
    background-color: #252526;
}
QLabel#KernelViz {
    border: 1px solid #3c3c3c;
    background-color: #000;
}
"""

class ImageViewer(QGraphicsView):
    """Custom QGraphicsView to support smooth panning and zooming."""
    def __init__(self, title=""):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag) # Click and drag to pan
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 1.0
        self._pixmap_item = None
        
        # Title overlay
        self.title = title
        self.setMouseTracking(True)

    def set_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene().addItem(self._pixmap_item)
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        else:
            self._pixmap_item.setPixmap(pixmap)

    def wheelEvent(self, event):
        # Mouse wheel to zoom
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom *= factor
        else:
            factor = 0.8
            self._zoom *= factor
            
        # Clamp zoom
        if self._zoom < 0.1:
            self._zoom = 0.1
        elif self._zoom > 10.0:
            self._zoom = 10.0
            
        self.setZoom(self._zoom)

    def setZoom(self, zoom):
        self._zoom = zoom
        # Reset transform and apply new scale
        self.resetTransform()
        self.scale(self._zoom, self._zoom)

    def reset_view(self):
        self._zoom = 1.0
        if self._pixmap_item:
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def get_pixel_color(self, pos):
        """Maps a viewport position to scene image pixel coordinates."""
        if not self._pixmap_item:
            return None, None, None, None
        scene_pos = self.mapToScene(pos)
        x, y = int(scene_pos.x()), int(scene_pos.y())
        
        img = self._pixmap_item.pixmap().toImage()
        if 0 <= x < img.width() and 0 <= y < img.height():
            pixel = img.pixel(x, y)
            r = (pixel >> 16) & 0xff
            g = (pixel >> 8) & 0xff
            b = pixel & 0xff
            return x, y, (r, g, b), img.size()
        return None, None, None, None


class ConvolutionVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Convolution Studio")
        self.resize(1300, 750)
        self.setStyleSheet(STYLE_SHEET)
        
        self.original_image = None
        self.kernel_size = 3
        
        # States
        self.is_grayscale = False
        self.normalize_output = True
        self.abs_output = False
        self.heatmap = False
        self.invert_img = False
        self.normalize_kernel = False
        self.true_convolution = False
        self.diff_map = False
        
        self.init_ui()
        self.load_default_image()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        
        # --- Left Panel (Images) ---
        image_layout = QVBoxLayout()
        
        viewer_layout = QHBoxLayout()
        
        self.viewer_original = ImageViewer("Original")
        self.viewer_filtered = ImageViewer("Filtered")
        self.viewer_filtered.setMouseTracking(True)
        self.viewer_filtered.viewport().installEventFilter(self)

        viewer_layout.addWidget(self.viewer_original, 50)
        viewer_layout.addWidget(self.viewer_filtered, 50)
        image_layout.addLayout(viewer_layout)

        # Sync views button
        sync_layout = QHBoxLayout()
        self.btn_reset_view = QPushButton("Reset Zoom & Pan")
        self.btn_reset_view.clicked.connect(self.reset_viewers)
        sync_layout.addStretch()
        sync_layout.addWidget(self.btn_reset_view)
        image_layout.addLayout(sync_layout)

        # --- Right Panel (Controls) ---
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(10)
        
        # File IO
        io_group = QGroupBox("File")
        io_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_save = QPushButton("Save Image")
        self.btn_save.clicked.connect(self.save_image)
        io_layout.addWidget(self.btn_load)
        io_layout.addWidget(self.btn_save)
        io_group.setLayout(io_layout)
        controls_layout.addWidget(io_group)

        # Image Processing Options
        img_proc_group = QGroupBox("Pre & Post Processing")
        img_proc_layout = QVBoxLayout()
        
        self.chk_gray = QCheckBox("Process as Grayscale")
        self.chk_gray.stateChanged.connect(self.update_states)
        img_proc_layout.addWidget(self.chk_gray)

        self.chk_invert = QCheckBox("Invert Image (Pre-processing)")
        self.chk_invert.stateChanged.connect(self.update_states)
        img_proc_layout.addWidget(self.chk_invert)

        self.chk_heatmap = QCheckBox("Apply False Color Heatmap (Post)")
        self.chk_heatmap.stateChanged.connect(self.update_states)
        img_proc_layout.addWidget(self.chk_heatmap)
        
        self.chk_diff = QCheckBox("Show Difference Map (Original vs Filtered)")
        self.chk_diff.stateChanged.connect(self.update_states)
        img_proc_layout.addWidget(self.chk_diff)
        
        img_proc_group.setLayout(img_proc_layout)
        controls_layout.addWidget(img_proc_group)

        # Convolution Options
        conv_group = QGroupBox("Convolution Options")
        conv_layout = QVBoxLayout()
        
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Size:"))
        self.combo_size = QComboBox()
        self.combo_size.addItems(["3x3", "5x5", "7x7"])
        self.combo_size.currentIndexChanged.connect(self.change_kernel_size)
        settings_layout.addWidget(self.combo_size)
        
        settings_layout.addWidget(QLabel("Border:"))
        self.combo_border = QComboBox()
        self.combo_border.addItems(["Reflect", "Replicate", "Zero", "Wrap"])
        self.combo_border.currentIndexChanged.connect(self.apply_filter)
        settings_layout.addWidget(self.combo_border)
        settings_layout.addStretch()
        conv_layout.addLayout(settings_layout)

        math_layout = QHBoxLayout()
        self.chk_norm_kern = QCheckBox("Auto-Normalize Kernel")
        self.chk_norm_kern.stateChanged.connect(self.apply_filter)
        math_layout.addWidget(self.chk_norm_kern)
        
        self.chk_true_conv = QCheckBox("True Convolution (Flip Kernel)")
        self.chk_true_conv.stateChanged.connect(self.update_states)
        math_layout.addWidget(self.chk_true_conv)
        conv_layout.addLayout(math_layout)

        math_layout2 = QHBoxLayout()
        self.chk_abs = QCheckBox("Absolute Value")
        self.chk_abs.stateChanged.connect(self.update_states)
        math_layout2.addWidget(self.chk_abs)
        
        self.chk_norm = QCheckBox("Normalize Output (0-255)")
        self.chk_norm.setChecked(True)
        self.chk_norm.stateChanged.connect(self.update_states)
        math_layout2.addWidget(self.chk_norm)
        conv_layout.addLayout(math_layout2)
        
        conv_group.setLayout(conv_layout)
        controls_layout.addWidget(conv_group)

        # Presets
        preset_group = QGroupBox("Presets (3x3 Only)")
        preset_layout = QHBoxLayout()
        self.combo_presets = QComboBox()
        self.combo_presets.addItems([
            "Custom", "Identity", "Box Blur", "Gaussian Blur", 
            "Sharpen", "Sobel X", "Sobel Y", "Laplacian", "Outline", "Emboss"
        ])
        self.combo_presets.currentIndexChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.combo_presets)
        preset_group.setLayout(preset_layout)
        controls_layout.addWidget(preset_group)

        # Kernel Grid & Viz
        kernel_group = QGroupBox("Kernel Matrix")
        kernel_v_layout = QHBoxLayout()
        
        # Left side: Matrix inputs
        matrix_container = QVBoxLayout()
        quick_btn_layout = QHBoxLayout()
        btn_zero = QPushButton("Zero Out")
        btn_zero.clicked.connect(lambda: self.set_matrix_all(0.0))
        btn_identity = QPushButton("Set Identity")
        btn_identity.clicked.connect(self.set_identity)
        matrix_container.addLayout(quick_btn_layout)
        
        self.kernel_grid_layout = QGridLayout()
        self.kernel_grid_layout.setSpacing(5)
        self.kernel_inputs = []
        self.build_kernel_grid(self.kernel_size)
        matrix_container.addLayout(self.kernel_grid_layout)
        
        kernel_v_layout.addLayout(matrix_container, 70)

        # Right side: Visual Heatmap of Kernel
        viz_container = QVBoxLayout()
        viz_container.addWidget(QLabel("Visual:"))
        self.lbl_kernel_viz = QLabel()
        self.lbl_kernel_viz.setObjectName("KernelViz")
        self.lbl_kernel_viz.setMinimumSize(80, 80)
        self.lbl_kernel_viz.setAlignment(Qt.AlignCenter)
        viz_container.addWidget(self.lbl_kernel_viz)
        
        # Kernel Import/Export
        io_kern_layout = QHBoxLayout()
        self.btn_save_kern = QPushButton("Save")
        self.btn_save_kern.clicked.connect(self.save_kernel)
        self.btn_load_kern = QPushButton("Load")
        self.btn_load_kern.clicked.connect(self.load_kernel)
        io_kern_layout.addWidget(self.btn_save_kern)
        io_kern_layout.addWidget(self.btn_load_kern)
        viz_container.addLayout(io_kern_layout)
        
        kernel_v_layout.addLayout(viz_container, 30)
        kernel_group.setLayout(kernel_v_layout)
        controls_layout.addWidget(kernel_group)

        # Info Label
        self.lbl_info = QLabel("Kernel Sum: 1.00 | Effective Sum: 1.00")
        self.lbl_info.setStyleSheet("font-weight: bold; color: #569cd6;")
        controls_layout.addWidget(self.lbl_info)

        controls_layout.addStretch()
        
        top_layout.addLayout(image_layout, 70)
        top_layout.addLayout(controls_layout, 30)
        main_layout.addLayout(top_layout, 95)

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready. Scroll to zoom, click-drag to pan.")
        main_layout.addWidget(self.status_bar)

    def build_kernel_grid(self, size):
        for i in reversed(range(self.kernel_grid_layout.count())): 
            widget = self.kernel_grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        self.kernel_inputs = []
        for i in range(size):
            row = []
            for j in range(size):
                spin = QDoubleSpinBox()
                spin.setRange(-100.0, 100.0)
                spin.setSingleStep(0.1)
                spin.setDecimals(2)
                spin.setAlignment(Qt.AlignCenter)
                spin.valueChanged.connect(self.apply_filter)
                self.kernel_grid_layout.addWidget(spin, i, j)
                row.append(spin)
            self.kernel_inputs.append(row)

    def set_matrix_all(self, value):
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.kernel_inputs[i][j].setValue(value)

    def set_identity(self):
        self.set_matrix_all(0.0)
        mid = self.kernel_size // 2
        self.kernel_inputs[mid][mid].setValue(1.0)

    def change_kernel_size(self, index):
        sizes = [3, 5, 7]
        self.kernel_size = sizes[index]
        self.build_kernel_grid(self.kernel_size)
        self.set_identity()
        self.combo_presets.setCurrentIndex(0)
        self.apply_filter()

    def update_states(self):
        self.is_grayscale = self.chk_gray.isChecked()
        self.normalize_output = self.chk_norm.isChecked()
        self.abs_output = self.chk_abs.isChecked()
        self.heatmap = self.chk_heatmap.isChecked()
        self.invert_img = self.chk_invert.isChecked()
        self.true_convolution = self.chk_true_conv.isChecked()
        self.diff_map = self.chk_diff.isChecked()
        self.apply_filter()

    def load_default_image(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
        cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)
        cv2.line(img, (50, 350), (350, 250), (0, 0, 255), 5)
        cv2.putText(img, "Convolution", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.original_image = img
        self.viewer_original.set_image(img)
        self.apply_preset(1)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if file_name:
            img = cv2.imread(file_name)
            h, w = img.shape[:2]
            max_size = 800
            if max(h, w) > max_size:
                scaling = max_size / float(max(h, w))
                img = cv2.resize(img, (int(w * scaling), int(h * scaling)))
            self.original_image = img
            self.viewer_original.set_image(img)
            self.reset_viewers()
            self.status_bar.showMessage(f"Loaded: {file_name.split('/')[-1]} ({w}x{h})")
            self.apply_filter()

    def save_image(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "filtered.png", "PNG (*.png);;JPEG (*.jpg)")
        if file_name:
            processed_img = self.process_image()
            if processed_img is not None:
                cv2.imwrite(file_name, processed_img)

    def save_kernel(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Kernel", "kernel.npy", "Numpy Array (*.npy)")
        if file_name:
            kernel = self.get_raw_kernel()
            np.save(file_name, kernel)

    def load_kernel(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Kernel", "", "Numpy Array (*.npy)")
        if file_name:
            kernel = np.load(file_name)
            size = kernel.shape[0]
            sizes = [3, 5, 7]
            if size in sizes and size == kernel.shape[1]:
                self.combo_size.setCurrentIndex(sizes.index(size))
                for i in range(size):
                    for j in range(size):
                        self.kernel_inputs[i][j].blockSignals(True)
                        self.kernel_inputs[i][j].setValue(float(kernel[i, j]))
                        self.kernel_inputs[i][j].blockSignals(False)
                self.apply_filter()
            else:
                self.status_bar.showMessage("Error: Kernel must be 3x3, 5x5, or 7x7.")

    def apply_preset(self, index):
        if self.kernel_size != 3:
            self.combo_size.setCurrentIndex(0)
            return
        presets = {
            0: None, 
            1: [0,0,0, 0,1,0, 0,0,0],       
            2: [1,1,1, 1,1,1, 1,1,1],       
            3: [1,2,1, 2,4,2, 1,2,1],       
            4: [0,-1,0, -1,5,-1, 0,-1,0],   
            5: [-1,0,1, -2,0,2, -1,0,1],    
            6: [-1,-2,-1, 0,0,0, 1,2,1],    
            7: [0,1,0, 1,-4,1, 0,1,0],      
            8: [-1,-1,-1, -1,8,-1, -1,-1,-1], 
            9: [-2,-1,0, -1,1,1, 0,1,2]     
        }
        vals = presets[index]
        if vals:
            for i in range(3):
                for j in range(3):
                    self.kernel_inputs[i][j].blockSignals(True)
                    self.kernel_inputs[i][j].setValue(vals[i*3 + j])
                    self.kernel_inputs[i][j].blockSignals(False)
            self.apply_filter()

    def get_raw_kernel(self):
        kernel = []
        for i in range(self.kernel_size):
            row = [self.kernel_inputs[i][j].value() for j in range(self.kernel_size)]
            kernel.append(row)
        return np.array(kernel, dtype=np.float32)

    def update_kernel_viz(self, kernel):
        """Creates a visual heatmap representation of the kernel matrix."""
        h, w = kernel.shape
        max_abs = np.max(np.abs(kernel))
        if max_abs == 0: max_abs = 1
        
        # Normalize to -1.0 to 1.0
        norm_k = kernel / max_abs
        
        viz_img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                val = norm_k[i, j]
                if val > 0:
                    viz_img[i, j] = [0, int(val * 255), 0]  # Green for positive
                elif val < 0:
                    viz_img[i, j] = [0, 0, int(abs(val) * 255)]  # Red for negative
        
        # Scale up to 80x80 using nearest neighbor
        viz_img_scaled = cv2.resize(viz_img, (80, 80), interpolation=cv2.INTER_NEAREST)
        viz_img_scaled = cv2.cvtColor(viz_img_scaled, cv2.COLOR_RGB2BGR)
        
        # Display in label
        q_img = QImage(viz_img_scaled.data, 80, 80, 3*80, QImage.Format_RGB888)
        self.lbl_kernel_viz.setPixmap(QPixmap.fromImage(q_img))

    def process_image(self):
        if self.original_image is None: return None

        img = self.original_image.copy()
        if self.invert_img: img = cv2.bitwise_not(img)
        if self.is_grayscale: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        raw_kernel = self.get_raw_kernel()
        self.update_kernel_viz(raw_kernel)
        kernel = raw_kernel.copy()
        
        # Kernel Auto-Normalization
        effective_sum = np.sum(raw_kernel)
        if self.chk_norm_kern.isChecked() and effective_sum != 0:
            kernel = kernel / effective_sum

        # True Convolution flips the kernel
        if self.true_convolution:
            kernel = cv2.flip(kernel, -1)

        border_map = {
            "Reflect": cv2.BORDER_REFLECT_101,
            "Replicate": cv2.BORDER_REPLICATE,
            "Zero": cv2.BORDER_CONSTANT,
            "Wrap": cv2.BORDER_WRAP
        }
        border_mode = border_map[self.combo_border.currentText()]

        # Convolution (CV_64F prevents clipping)
        filtered = cv2.filter2D(img, cv2.CV_64F, kernel, borderType=border_mode)
        
        # Difference Map
        if self.diff_map:
            # Convert original to same depth (64F) for subtraction
            img_64f = img.astype(np.float64)
            filtered = cv2.absdiff(img_64f, filtered)

        if self.abs_output:
            filtered = np.abs(filtered)
        if self.normalize_output:
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
        
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

        if self.heatmap:
            if len(filtered.shape) == 3:
                filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            filtered = cv2.applyColorMap(filtered, cv2.COLORMAP_JET)

        if len(filtered.shape) == 2:
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

        self.lbl_info.setText(f"Kernel Sum: {np.sum(raw_kernel):.2f} | Effective Sum: {np.sum(kernel):.2f}")
        return filtered

    def apply_filter(self):
        if self.original_image is None: return
        filtered_img = self.process_image()
        if filtered_img is not None:
            self.viewer_filtered.set_image(filtered_img)

    def reset_viewers(self):
        self.viewer_original.reset_view()
        self.viewer_filtered.reset_view()

    def eventFilter(self, source, event):
        if source == self.viewer_filtered.viewport() and event.type() == QEvent.MouseMove:
            pos = event.pos()
            x, y, rgb, size = self.viewer_filtered.get_pixel_color(pos)
            if x is not None:
                self.status_bar.showMessage(f"Image Size: {size.width()}x{size.height()} | Pixel [X={x}, Y={y}] -> RGB{rgb} | Zoom: {self.viewer_filtered._zoom:.2f}x")
        return super().eventFilter(source, event)

    def resizeEvent(self, event):
        # Keep images fitted when window resizes if zoom is 1.0
        if self.viewer_original._zoom <= 1.0:
            self.viewer_original.reset_view()
        if self.viewer_filtered._zoom <= 1.0:
            self.viewer_filtered.reset_view()
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConvolutionVisualizer()
    window.show()
    sys.exit(app.exec())