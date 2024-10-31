import numpy
import numpy as np


def manual(image, threshold):
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 2:
        imgthresh = np.where(image >= threshold, 255, 0)
    else:
        imgthresh = np.zeros_like(image)
        for i in range(3):
            imgthresh[:, :, i] = np.where(image[:, :, i] >= threshold, 255, 0)
    print("inside manual", imgthresh.dtype)
    return imgthresh.astype(np.float32) / 255.0


def automatic(image, initial_threshold=128, epsilon=1):
    """
    Perform automatic thresholding on an image.
    Works for both grayscale and RGB images.
    Returns a binary image and a list of threshold values for each channel.
    """
    # Ensure the image values are within the 0-255 range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    thresholds = []
    if len(image.shape) == 2:  # Grayscale image
        threshold = compute_threshold(image, initial_threshold, epsilon)
        thresholds.append(threshold)
        binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)
        return binary_image, thresholds

    else:  # RGB image
        binary_image = np.zeros_like(image, dtype=np.uint8)

        # Apply threshold independently to each channel
        for i in range(3):
            channel = image[:, :, i]
            threshold = compute_threshold(channel, initial_threshold, epsilon)
            thresholds.append(threshold)
            binary_image[:, :, i] = np.where(channel > threshold, 255, 0)
        return binary_image.astype(np.float32) / 255.0, thresholds


def compute_threshold(channel, initial_threshold=128, epsilon=1):
    """
    Compute the optimal threshold for a single channel using an iterative method.
    """
    T = initial_threshold  # Start with an initial guess

    while True:
        # Split the channel into two groups based on the current threshold
        G1 = channel[channel <= T]
        G2 = channel[channel > T]

        # Check if groups are empty and set means to 0 if so
        mu1 = np.mean(G1) if G1.size > 0 else T
        mu2 = np.mean(G2) if G2.size > 0 else T

        # Update the threshold
        T_new = (mu1 + mu2) / 2

        # Check for convergence
        if abs(T - T_new) < epsilon:
            break

        T = T_new

    return T