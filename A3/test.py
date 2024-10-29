import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton

def plot_histograms(image, ax):
    ax.clear()
    if len(image.shape) == 2:  # Grayscale image
        ax.hist(image.ravel(), range(256), color='gray', alpha=0.5, histtype='step', label='Grayscale', density=True)
    else:  # Color image (3 channels)
        r, g, b = image[:, :, 0] * 255, image[:, :, 1] * 255, image[:, :, 2] * 255
        ax.hist(r.ravel(), bins=range(256), color='r', alpha=0.5, histtype='step', label='Red Channel', density=True)
        ax.hist(g.ravel(), bins=range(256), color='g', alpha=0.5, histtype='step', label='Green Channel', density=True)
        ax.hist(b.ravel(), bins=range(256), color='b', alpha=0.5, histtype='step', label='Blue Channel', density=True)

def update_image(ax_img, ax_hist, img):
    ax_img.clear()
    ax_img.imshow(img, cmap='Greys_r')
    ax_img.axis('off')
    plot_histograms(img, ax_hist)
    plt.draw()

def open_file_explorer(event):
    global image
    app = QApplication(sys.argv)
    file_dialog = QFileDialog()
    image_path, _ = file_dialog.getOpenFileName(None, "Select Image", "", "Images (*.png *.jpg *.bmp)")
    if image_path:
        image = mpimg.imread(image_path)
        update_image(ax_img1, ax_hist, image)
    app.exit()

class ThresholdDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Threshold Values")

        layout = QVBoxLayout()
        self.red_input = QLineEdit(self)
        self.green_input = QLineEdit(self)
        self.blue_input = QLineEdit(self)

        layout.addWidget(QLabel("Red Threshold:"))
        layout.addWidget(self.red_input)
        layout.addWidget(QLabel("Green Threshold:"))
        layout.addWidget(self.green_input)
        layout.addWidget(QLabel("Blue Threshold:"))
        layout.addWidget(self.blue_input)

        submit_button = QPushButton("Apply", self)
        submit_button.clicked.connect(self.accept)
        layout.addWidget(submit_button)

        self.setLayout(layout)

    def get_values(self):
        """Retrieve threshold values or return None if input is invalid."""
        try:
            r = int(self.red_input.text())
            g = int(self.green_input.text())
            b = int(self.blue_input.text())
            return r, g, b
        except ValueError:
            return None  # Invalid input

def open_threshold_dialog():
    """Open the threshold dialog and return user input values."""
    app = QApplication(sys.argv)
    dialog = ThresholdDialog()
    values = None
    if dialog.exec_() == QDialog.Accepted:
        values = dialog.get_values()
    app.exit()
    return values

def apply_threshold(event):
    """Handle the threshold application logic."""
    values = open_threshold_dialog()
    if values:
        print(f"Thresholds - Red: {values[0]}, Green: {values[1]}, Blue: {values[2]}")
        # TODO: Apply thresholding logic here using the retrieved values
    else:
        print("Invalid input. Please enter valid integer values for all thresholds.")

# Initialize default image
image = np.zeros((256, 256, 3), dtype=np.uint8)

fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1])

ax_img1 = fig.add_subplot(gs[0, 0])
ax_img1.imshow(image, cmap='Greys_r')
ax_img1.set_title('Input Image')
ax_img1.axis('off')

ax_img2 = fig.add_subplot(gs[0, 2])
ax_img2.imshow(image, cmap='Greys_r')
ax_img2.set_title('Output Image')
ax_img2.axis('off')

ax_button_open = plt.axes([0.1, 0.05, 0.15, 0.075])
ax_button_thresh = plt.axes([0.35, 0.05, 0.15, 0.075])

button_open_file = Button(ax_button_open, 'Open Image')
button_threshold = Button(ax_button_thresh, 'Set Thresholds')

button_open_file.on_clicked(open_file_explorer)
button_threshold.on_clicked(apply_threshold)

ax_hist = fig.add_subplot(gs[0, 1])
plot_histograms(image, ax_hist)

plt.tight_layout()
plt.show()