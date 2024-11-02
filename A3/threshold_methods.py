import numpy as np

def meanc(image, offset=0):

    window_size = 7
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 2:
        binary_image = apply_adaptive_threshold(image, window_size, offset)
        return binary_image

    else:
        binary_image = np.zeros_like(image, dtype=np.uint8)

        for i in range(3):
            channel = image[:, :, i]
            binary_image[:, :, i] = apply_adaptive_threshold(channel, window_size, offset)

        return binary_image.astype(np.float32) / 255.0


def apply_adaptive_threshold(channel, window_size, offset):
    """
    Apply adaptive thresholding to a single channel using a local mean and offset.
    """
    pad_size = window_size // 2
    padded_channel = np.pad(channel, pad_size, mode='reflect')
    binary_channel = np.zeros_like(channel, dtype=np.uint8)

    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            # Extract the 7x7 neighborhood around the pixel
            local_region = padded_channel[i:i + window_size, j:j + window_size]
            local_mean = np.mean(local_region)

            # Apply threshold
            if channel[i, j] > (local_mean - offset):
                binary_channel[i, j] = 255
            else:
                binary_channel[i, j] = 0

    return binary_channel

def otsu(image):
    """
    Perform Otsu's thresholding on an image.
    Returns a binary image and a list of threshold values for each channel.
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    thresholds = []
    if len(image.shape) == 2:
        threshold = compute_otsu_threshold(image)
        thresholds.append(threshold)
        binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)
        return binary_image, thresholds

    else:
        binary_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(3):
            channel = image[:, :, i]
            threshold = compute_otsu_threshold(channel)
            thresholds.append(int(threshold))
            binary_image[:, :, i] = np.where(channel > threshold, 255, 0)
        return binary_image.astype(np.float32) / 255.0, thresholds


def compute_otsu_threshold(channel):
    """
    Compute the threshold for a single channel using Otsu's method.
    """

    histogram, _ = np.histogram(channel, bins=256, range=(0, 256))
    total_pixels = channel.size

    max_variance = 0
    optimal_threshold = 0
    cumulative_prob = np.cumsum(histogram) / total_pixels
    cumulative_mean = np.cumsum(histogram * np.arange(256)) / total_pixels

    global_mean = cumulative_mean[-1]

    for t in range(256):
        weight_bg = cumulative_prob[t]  # Weight for background
        weight_fg = 1 - weight_bg  # Weight for foreground

        if weight_bg == 0 or weight_fg == 0:
            continue

        mean_bg = cumulative_mean[t] / weight_bg
        mean_fg = (global_mean - cumulative_mean[t]) / weight_fg

        variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance_between > max_variance:
            max_variance = variance_between
            optimal_threshold = t

    return optimal_threshold

def manual(image, threshold):
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 2:
        imgthresh = np.where(image >= threshold, 255, 0)
    else:
        imgthresh = np.zeros_like(image)
        for i in range(3):
            imgthresh[:, :, i] = np.where(image[:, :, i] >= threshold, 255, 0)

    return imgthresh.astype(np.float32) / 255.0

def automatic(image, initial_threshold=128, epsilon=1):
    """
    Perform automatic thresholding on an image.
    Returns a binary image and a list of threshold values for each channel.
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    thresholds = []
    if len(image.shape) == 2:
        threshold = compute_threshold(image, initial_threshold, epsilon)
        thresholds.append(threshold)
        binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)
        return binary_image, thresholds

    else:
        binary_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(3):
            channel = image[:, :, i]
            threshold = compute_threshold(channel, initial_threshold, epsilon)
            thresholds.append(threshold)
            binary_image[:, :, i] = np.where(channel > threshold, 255, 0)
        return binary_image.astype(np.float32) / 255.0, thresholds


def compute_threshold(channel, initial_threshold=128, epsilon=1):
    """
    Compute the threshold for a single channel using iterative method.
    """
    T = initial_threshold

    while True:
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