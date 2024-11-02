"""
Name: Léo Gilbert
Student #: 202038196

This program provides a user interface, which allows the user to perform Manual Thresholding with user provided
threshold value, Automatic Threshold based on image histogram, Otsu's Thresholding method, and Adaptive Mean-C
Thresholding. The program contains buttons for these operations, a button for opening an image as well as text boxes
for the user supplied threshold value (Manual) or offset value (Adaptive Mean-C). All the thresholding related functions
can be found in threshold_methods.py, and main.py is responsible for the gui and buttons.

The code uses MatPlotLib, but only for gui related tasks, not for any of the actual image processing.

run the following commands to install dependencies:
pip install pyqt5
pip install matplotlib

*BUGS*
-HoughCircles3.png would not load correctly
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PyQt5.QtWidgets import QApplication, QFileDialog
import threshold_methods

COLOURS = ['Red', 'Green', 'Blue']

def btn_meanc(event=None):
    offset = slider_offset.val
    imgthresh = threshold_methods.meanc(image, offset)
    update_image(ax_img2, ax_hist, imgthresh)

def btn_otsu(event):
    imgthresh, thresholds = threshold_methods.otsu(image)
    update_image(ax_img2, ax_hist, imgthresh)
    lines = []
    if len(thresholds) > 1:
        for i in range(len(imgthresh.shape)):
            thresh_line = plt.vlines(x=thresholds[i], ymin=0, ymax=1, colors='green', ls=':', lw=1,
                                     label=f'Threshold Value {COLOURS[i]} - {thresholds[i]}')
            lines.append(thresh_line)
    else:
        thresh_line = plt.vlines(x=thresholds[0], ymin=0, ymax=1, colors='green', ls=':', lw=1,
                                 label=f'Threshold Value - {round(thresholds[0], 2)}')
        lines.append(thresh_line)
    ax_hist.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 1.2))

def btn_automatic(event):
    imgthresh, thresholds = threshold_methods.automatic(image)
    update_image(ax_img2, ax_hist, imgthresh)
    lines = []
    if len(thresholds) > 1:
        for i in range(len(imgthresh.shape)):
            thresh_line = plt.vlines(x=thresholds[i], ymin=0, ymax=1, colors='green', ls=':', lw=1,
                                     label=f'Threshold Value {COLOURS[i]} - {round(thresholds[i], 2)}')
            lines.append(thresh_line)
    else:
        thresh_line = plt.vlines(x=thresholds[0], ymin=0, ymax=1, colors='green', ls=':', lw=1,
                                 label=f'Threshold Value - {round(thresholds[0], 2)}')
        lines.append(thresh_line)
    ax_hist.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 1.2))

def btn_manual(event):
    thresh_val = text_threshold.text
    try:
        thresh = int(thresh_val)
        imgthresh = threshold_methods.manual(image, thresh)
        update_image(ax_img2, ax_hist, imgthresh)
        thresh_line = plt.vlines(x=thresh, ymin=0, ymax=1, colors='green', ls=':', lw=1, label='Threshold Value')
        ax_hist.legend(handles=[thresh_line], loc='upper center', bbox_to_anchor=(0.5, 1.2))
    except ValueError:
        print("Invalid sigma value. Please enter a valid number.")

def plot_histograms(image, ax):
    ax.clear()
    if len(image.shape) == 2:  # Grayscale image
        ax.hist(image.ravel(), range(256), color='gray', alpha=0.5, histtype='step', label='Grayscale', density=True)
    else:  # Color image (3 channels)
        r = image[:, :, 0] * 255
        g = image[:, :, 1] * 255
        b = image[:, :, 2] * 255
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
        print(image.dtype)
    app.exit()

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

ax_button1 = plt.axes([0.6, 0.25, 0.15, 0.075])
ax_button2 = plt.axes([0.35, 0.05, 0.15, 0.075])
ax_button3 = plt.axes([0.6, 0.05, 0.15, 0.075])
ax_button4 = plt.axes([0.1, 0.05, 0.15, 0.075])
axbox = plt.axes([0.1, 0.25, 0.15, 0.075])
ax_slider = plt.axes([0.82, 0.05, 0.15, 0.03])  # Define slider position
ax_button5 = plt.axes([0.35, 0.25, 0.15, 0.075])

button_open_file = Button(ax_button4, 'Open Image')
button_automatic = Button(ax_button1, 'Automatic Threshold')
button_otsu = Button(ax_button2, "Otsu's Method")
button_adaptive = Button(ax_button3, 'Adaptive Thresholding\n(7x7)')
button_manual = Button(ax_button5, 'Manual Threshold')
text_threshold = TextBox(axbox, 'Threshold\nValue', initial="128")

# Define the offset slider
slider_offset = Slider(ax_slider, 'Offset', -50, 50, valinit=0, valstep=1)

button_open_file.on_clicked(open_file_explorer)
button_manual.on_clicked(btn_manual)
button_automatic.on_clicked(btn_automatic)
button_otsu.on_clicked(btn_otsu)
button_adaptive.on_clicked(btn_meanc)

# Update the image every time the slider is changed
slider_offset.on_changed(btn_meanc)

ax_hist = fig.add_subplot(gs[0, 1])
plot_histograms(image, ax_hist)

plt.tight_layout()
plt.show()