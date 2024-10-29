"""
Name: Léo Gilbert
Student #: 202038196

This program provides a user interface, which allows the user to perform stretch, aggressive stretch, and equalization
on any of the sample images provided. I completed all parts of the code myself.

In general, each image processing operation is split amongst a few functions (2-3).

*BUGS*
-Stretch and aggressive stretch do not behave correctly after equalization is used on the image,
image must be reloaded for correct behaviour
-When clicking between the regular histogram (the 'Display Histogram' button and any of the other options, you must
click the 'Display Histogram' button twice in order for the histogram to show.
-HoughCircles3.png would not load correctly
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import colorsys
from PyQt5.QtWidgets import QApplication, QFileDialog


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
    ax_img1.set_title('Input Image')
    ax_img2.set_title('Output Image')
    plot_histograms(img, ax_hist)
    plt.draw()


def btn_histogram(event):
    plot_histograms(image, ax_hist)
    update_image(image, ax_hist, image)
def btn_histogram_stretch(event):
    stretched_img = np.rint(histogram_stretch(image)).astype(np.uint8)
    update_image(ax_img2, ax_hist, stretched_img)

def btn_histogram_agr_stretch(event):
    try:
        cutoff_fraction = float(text_box.text)
        agr_stretched_img = aggressive_histogram_stretch(image, cutoff_fraction)
        update_image(ax_img2, ax_hist, agr_stretched_img)
    except ValueError:
        print("Invalid cutoff fraction. Please enter a valid number.")

def btn_histogram_equilization(event):
    equalized_img = equalize_histogram(image)
    update_image(ax_img2, ax_hist, equalized_img)

def histogram_stretch(image):
    if len(image.shape) == 2:
        return stretch_channel(image)
    elif len(image.shape) == 3:
        stretched = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            stretched[:, :, i] = stretch_channel(image[:, :, i])
        return np.clip(np.rint(stretched), 0, 255).astype(np.uint8)

def stretch_channel(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = 255 * ((image - min_val) / (max_val - min_val))
    unique_vals = np.unique(stretched)
    stretched = np.interp(stretched, unique_vals, np.linspace(0, 255, len(unique_vals)))
    return stretched

def aggressive_histogram_stretch(image, cutoff_fraction):
    if len(image.shape) == 2:
        p1, p99 = np.percentile(image, (cutoff_fraction, 100 - cutoff_fraction))
        stretched = np.clip(((image - p1) / (p99 - p1)) * 255, 0, 255)
        return stretched.astype(np.uint8)

    elif len(image.shape) == 3:
        stretched = np.zeros_like(image)
        for i in range(3):
            p1, p99 = np.percentile(image[:, :, i], (cutoff_fraction, 100 - cutoff_fraction))
            stretched[:, :, i] = np.clip(((image[:, :, i] - p1) / (p99 - p1)) * 255, 0, 255)
        return stretched.astype(np.uint8)

def equalize_histogram(image):
    if len(image.shape) == 2:
        return equalize(image)
    elif len(image.shape) == 3:
        rgb_to_hsl(image)
        return equalize(image)

def equalize(image):
    if len(image.shape) == 2:
        l = (image[:,:].ravel() * 255).astype(int)
        histogram, _, _ = plt.hist(l, bins=range(257))
        cdf = histogram.cumsum()
        cdf_norm = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        l_eq = cdf_norm[l]
        image_eq = l_eq.reshape(image.shape) / 255
        return image_eq
    else:
        l = (image[:, :, 2].ravel() * 100).astype(int)
        histogram, _, _ = plt.hist(l, bins=range(102))
        cdf = histogram.cumsum()
        cdf_norm = (cdf - cdf.min()) * 100 / (cdf.max() - cdf.min())
        l_eq = cdf_norm[l]
        l_channel = l_eq.reshape(image[:, :, 2].shape) / 100
        image[:, :, 2] = l_channel
        return image

def rgb_to_hsl(image):
    hsl_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j] / 255.0
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            hsl_image[i, j] = [h, s, l]

    return hsl_image

def open_file_explorer(event):
    global image
    app = QApplication(sys.argv)
    file_dialog = QFileDialog()
    file_dialog.setDirectory("images")
    image_path, _ = file_dialog.getOpenFileName(None, "Select Image", "images", "Images (*.png *.jpg *.bmp)")
    if image_path:
        image = mpimg.imread(image_path)
        update_image(ax_img1, ax_hist, image)
    app.exit()

# Initialize a default image to prevent unresolved reference error
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

ax_button1 = plt.axes([0.1, 0.05, 0.15, 0.075])
ax_button2 = plt.axes([0.35, 0.05, 0.15, 0.075])
ax_button3 = plt.axes([0.6, 0.05, 0.15, 0.075])
ax_button4 = plt.axes([0.1, 0.25, 0.15, 0.075])
ax_button5 = plt.axes([0.35, 0.25, 0.15, 0.075])
axbox = plt.axes([0.82, 0.05, 0.15, 0.075])

button_histogram = Button(ax_button5, 'Display Histogram')
button_stretch = Button(ax_button1, label='Histogram Stretch')
button_agr_stretch = Button(ax_button2, label='Aggressive Stretch')
button_equilization = Button(ax_button3, label='Equalization')
button_open_file = Button(ax_button4, label='Open Image')
text_box = TextBox(axbox, 'Cutoff\nFraction', initial="0")


button_histogram.on_clicked((btn_histogram))
button_stretch.on_clicked(btn_histogram_stretch)
button_agr_stretch.on_clicked(btn_histogram_agr_stretch)
button_equilization.on_clicked(btn_histogram_equilization)
button_open_file.on_clicked(open_file_explorer)


ax_hist = fig.add_subplot(gs[0, 1])
plot_histograms(image, ax_hist)

plt.tight_layout()
plt.show()
