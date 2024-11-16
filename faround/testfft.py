import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Load a grayscale image
image = cv2.imread('image_with_noise.png', cv2.IMREAD_GRAYSCALE)

# Perform FFT on the image
f_transform = fft2(image)
f_transform_shifted = fftshift(f_transform)  # Shift zero frequency to center

# Display the magnitude spectrum
magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
plt.subplot(1, 2, 1)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum (Frequency Domain)")

# Create a mask to filter out noise (example: simple low-pass filter)
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1  # Low-pass filter mask

# Apply mask and inverse FFT
f_transform_shifted_filtered = f_transform_shifted * mask
f_ishift = ifftshift(f_transform_shifted_filtered)
img_back = ifft2(f_ishift)
img_back = np.abs(img_back)

# Display the filtered image
plt.subplot(1, 2, 2)
plt.imshow(img_back, cmap='gray')
plt.title("Filtered Image (Spatial Domain)")
plt.show()