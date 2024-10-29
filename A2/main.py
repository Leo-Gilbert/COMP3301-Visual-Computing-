"""
Name: Léo Gilbert
Student #: 202038196

This program provides a user interface, which allows the user to perform triangle, median, Gaussian, Kuwahara, and mean
filters and also provides a button for image selection, and a text box for the Sigma parameter used in the Gaussian
filter. All the filtering related functions can be found in filter.py, and main.py is responsible for the gui and
buttons.

The code uses MatPlotLib, but only for gui related tasks, not for any of the actual image processing.

run the following commands to install dependencies:
pip install pyqt5
pip install matplotlib
pip install numpy

*BUGS*
-HoughCircles3.png would not load correctly
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PyQt5.QtWidgets import QApplication, QFileDialog
import filter as ftr

def btn_mean_filter(event):
    global image
    image = ftr.mean_filter(image)
    update_image(ax_img2, image, title='Output Image')

def btn_kuwahara_filter(event):
    global image
    image = ftr.kuwahara_filter(image)
    update_image(ax_img2, image, title='Output Image')

def btn_gaussian_filter(event):
    global image
    sigma_value = text_sigma.text
    try:
        sigma = float(sigma_value)
        image = ftr.gaussian_filter(image, sigma=sigma)
        update_image(ax_img2, image, title='Output Image')
    except ValueError:
        print("Invalid sigma value. Please enter a valid number.")

def btn_median_filter(event):
    global image
    image = ftr.median_filter(image)
    update_image(ax_img2, image, title='Output Image')

def btn_triangle_filter(event):
    global image
    image = ftr.triangle_filter(image)
    update_image(ax_img2, image, title='Output Image')

def btn_add_noise(event):
    global image
    image = add_noise(image)
    update_image(ax_img2, image, title='Output Image')

def add_noise(image, mean=0, sigma=0.15):
    noise = np.random.normal(mean, sigma, image.shape)
    if image.max() <= 1.0:
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)
    else:
        noisy_image = image + (noise * 255)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def update_image(ax_img, img, title=None):
    ax_img.clear()
    ax_img.imshow(img, cmap='Greys_r')
    ax_img.axis('off')
    if title is not None:
        ax_img.set_title(title)
    plt.draw()

def open_file_explorer(event):
    global image
    app = QApplication(sys.argv)
    file_dialog = QFileDialog()
    file_dialog.setDirectory("images")
    image_path, _ = file_dialog.getOpenFileName(None, "Select Image", "images", "Images (*.png *.jpg *.bmp)")
    if image_path:
        image = mpimg.imread(image_path)
        update_image(ax_img1, image)
    app.exit()

image = np.zeros((256, 256, 3), dtype=np.uint8)
processed_image = np.zeros((256, 256, 3), dtype=np.uint8)

fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

ax_img1 = fig.add_subplot(gs[0, 0])
ax_img1.imshow(image, cmap='Greys_r')
ax_img1.set_title('Input Image')
ax_img1.axis('off')

ax_img2 = fig.add_subplot(gs[0, 1])
ax_img2.imshow(image, cmap='Greys_r')
ax_img2.set_title('Output Image')
ax_img2.axis('off')

ax_button1 = plt.axes([0.6, 0.25, 0.15, 0.075])
ax_button2 = plt.axes([0.35, 0.05, 0.15, 0.075])
ax_button3 = plt.axes([0.6, 0.05, 0.15, 0.075])
ax_button4 = plt.axes([0.1, 0.05, 0.15, 0.075])
ax_button6 = plt.axes([0.1, 0.25, 0.15, 0.075])
ax_button5 = plt.axes([0.82, 0.05, 0.15, 0.075])
ax_button7 = plt.axes([0.82, 0.25, 0.15, 0.075])
axbox = plt.axes([0.35, 0.25, 0.15, 0.075])

button_triangle = Button(ax_button1, 'Triangle Filter')
button_gaussian = Button(ax_button2, label='Gaussian Filter')
button_median = Button(ax_button3, label='Median Filter')
button_kuwahara = Button(ax_button5, label='Kuwahara Filter')
text_sigma = TextBox(axbox, 'Sigma', initial="1")
button_open_file = Button(ax_button4, label='Open Image')
button_noise = Button(ax_button6, label='Add Noise')
button_mean = Button(ax_button7, label='Mean Filter')

button_noise.on_clicked(btn_add_noise)
button_open_file.on_clicked(open_file_explorer)
button_triangle.on_clicked(btn_triangle_filter)
button_median.on_clicked(btn_median_filter)
button_gaussian.on_clicked(btn_gaussian_filter)
button_kuwahara.on_clicked(btn_kuwahara_filter)
button_mean.on_clicked(btn_mean_filter)

plt.tight_layout()
plt.show()