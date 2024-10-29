import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
import threshold_dialog
import threshold_methods

def btn_automatic(event):
    imgthresh = threshold_methods.automatic(image)
    update_image(ax_img2, ax_hist, imgthresh)
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
    print("inside update image", img.dtype)
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

def btn_otsu(event):
    r, g, b = open_threshold_dialog()
    print(r, g, b)

def open_threshold_dialog():
    app = QApplication(sys.argv)
    dialog = threshold_dialog.ThresholdDialog()
    if dialog.exec_() == QDialog.Accepted:
        values = dialog.get_values()
        if values:
            print(f"Thresholds - Red: {values[0]}, Green: {values[1]}, Blue: {values[2]}")
    return values
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
ax_button5 = plt.axes([0.35, 0.25, 0.15, 0.075])
#ax_button7 = plt.axes([0.82, 0.25, 0.15, 0.075])
#axbox = plt.axes([0.35, 0.25, 0.15, 0.075])

button_open_file = Button(ax_button4, 'Open Image')
button_automatic = Button(ax_button1, 'Automatic Threshold')
button_otsu = Button(ax_button2, "Otsu's Method")
button_adaptive = Button(ax_button3, 'Adaptive Thresholding\n(7x7)')
button_manual = Button(ax_button5, 'Manual Threshold')
text_threshold = TextBox(axbox, 'Threshold\nValue', initial="128")

button_open_file.on_clicked(open_file_explorer)
button_manual.on_clicked(btn_manual)
button_automatic.on_clicked(btn_automatic)
#button_threshold.on_clicked(open_threshold_dialog)

ax_hist = fig.add_subplot(gs[0, 1])
plot_histograms(image, ax_hist)

plt.tight_layout()
plt.show()