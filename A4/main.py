"""
Name: Léo Gilbert
Student #: 202038196

This program provides a user interface, which allows the user to convert color images to grayscale, use sobel edge
to detect edges, as well as draw lines, circles, and triangles. All image processing related helper functions can be
found in helpers.py

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
from matplotlib.widgets import Button
import matplotlib.image as mpimg
from PyQt5.QtWidgets import QApplication, QFileDialog
import helpers

def update_display(original_ax, transformed_ax, original_image, transformed_image):
    original_ax.clear()
    transformed_ax.clear()
    original_ax.imshow(original_image, cmap='gray')
    transformed_ax.imshow(transformed_image, cmap='gray')
    original_ax.set_title("Original Image")
    transformed_ax.set_title("Transformed Image")
    original_ax.axis('off')
    transformed_ax.axis('off')
    plt.draw()


def highlight_button(active_button):
    for btn in [button_line, button_triangle, button_circle]:
        btn.color = "white"
    active_button.color = "lightblue"
    plt.draw()


def on_grayscale_button(event):
    global current_image, transformed_image
    if current_image is not None:
        transformed_image = helpers.convert_to_grayscale(current_image)
        update_display(ax_original, ax_transformed, current_image, transformed_image)


def on_sobel_button(event):
    global transformed_image, edge_detected, blurred_image
    if transformed_image is not None:
        transformed_image = helpers.sobel_edge_detection(transformed_image)
        blurred_image = helpers.gaussian_filter(current_image.astype(float), sigma=3)
        edge_detected = True
        update_display(ax_original, ax_transformed, current_image, transformed_image)


def on_canvas_click(event):
    global click_points, transformed_image
    if event.inaxes == ax_transformed:
        click_points.append((int(event.xdata), int(event.ydata)))
        if mode == "line" and len(click_points) == 2:
            transformed_image = helpers.draw_line(transformed_image.copy(), click_points[0], click_points[1])
            update_display(ax_original, ax_transformed, current_image, transformed_image)
            click_points = []
        elif mode == "triangle" and len(click_points) == 3:
            transformed_image = helpers.draw_triangle(transformed_image.copy(), click_points[0], click_points[1], click_points[2])
            update_display(ax_original, ax_transformed, current_image, transformed_image)
            click_points = []
        elif mode == "circle" and len(click_points) == 2:
            radius = int(np.sqrt((click_points[1][0] - click_points[0][0])**2 + (click_points[1][1] - click_points[0][1])**2))
            transformed_image = helpers.draw_circle(transformed_image.copy(), click_points[0], radius)
            update_display(ax_original, ax_transformed, current_image, transformed_image)
            click_points = []


def on_line_button(event):
    global mode, click_points
    mode = "line"
    click_points = []
    highlight_button(button_line)


def on_triangle_button(event):
    global mode, click_points
    mode = "triangle"
    click_points = []
    highlight_button(button_triangle)


def on_circle_button(event):
    global mode, click_points
    mode = "circle"
    click_points = []
    highlight_button(button_circle)


def on_load_image(event):
    global current_image, transformed_image, edge_detected, blurred_image
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.jpeg)", options=options)
    if file_path:
        current_image = mpimg.imread(file_path)
        if current_image.dtype == np.float32:
            current_image = (current_image * 255).astype(np.uint8)
        transformed_image = current_image.copy()
        edge_detected = False
        blurred_image = None
        update_display(ax_original, ax_transformed, current_image, transformed_image)

# Main application setup
app = QApplication(sys.argv)

fig, (ax_original, ax_transformed) = plt.subplots(1, 2)
plt.subplots_adjust(bottom=0.3)
ax_original.axis('off')
ax_transformed.axis('off')

current_image = None
transformed_image = None
blurred_image = None
edge_detected = False
click_points = []
mode = None

fig.canvas.mpl_connect("button_press_event", on_canvas_click)

# Add buttons in two rows
# First row
ax_button_load = plt.axes([0.05, 0.15, 0.25, 0.075])
button_load = Button(ax_button_load, 'Load Image')
button_load.on_clicked(on_load_image)

ax_button_grayscale = plt.axes([0.35, 0.15, 0.25, 0.075])
button_grayscale = Button(ax_button_grayscale, 'Grayscale')
button_grayscale.on_clicked(on_grayscale_button)

ax_button_sobel = plt.axes([0.65, 0.15, 0.25, 0.075])
button_sobel = Button(ax_button_sobel, 'Sobel Edge')
button_sobel.on_clicked(on_sobel_button)

# Second row
ax_button_line = plt.axes([0.05, 0.05, 0.2, 0.075])
button_line = Button(ax_button_line, 'Line')
button_line.on_clicked(on_line_button)

ax_button_triangle = plt.axes([0.3, 0.05, 0.2, 0.075])
button_triangle = Button(ax_button_triangle, 'Triangle')
button_triangle.on_clicked(on_triangle_button)

ax_button_circle = plt.axes([0.55, 0.05, 0.2, 0.075])
button_circle = Button(ax_button_circle, 'Circle')
button_circle.on_clicked(on_circle_button)

plt.show()