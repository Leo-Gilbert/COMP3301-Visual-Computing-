import cv2
import numpy as np
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk


def detect_corners(file_path):
    """Detect corners in the selected image and display the result."""

    image = cv2.imread(file_path)
    if image is None:
        label_status.config(text="Error: Could not read image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    dst = cv2.dilate(dst, None)

    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow("Corners Detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def browse_file():
    """Open file browser to select an image."""
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")),
    )
    if file_path:
        label_status.config(text=f"Selected File: {file_path}")
        # Display the image in the GUI
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        label_image.config(image=img)
        label_image.image = img

        detect_corners(file_path)


root = Tk()
root.title("Corner Detection")
root.geometry("400x500")

label_status = Label(root, text="No file selected.")
label_status.pack(pady=10)

button_browse = Button(root, text="Browse Image", command=browse_file)
button_browse.pack(pady=10)

label_image = Label(root)
label_image.pack(pady=10)

root.mainloop()