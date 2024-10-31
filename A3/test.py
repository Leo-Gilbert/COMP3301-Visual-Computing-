import numpy as np
import matplotlib.pyplot as plt


def compute_threshold(channel, initial_threshold=128, epsilon=1):
    """
    Compute the optimal threshold for a single channel using the iterative mean method.
    """
    T = initial_threshold  # Initial threshold guess

    # Debugging: Initial condition
    print(f"\nStarting threshold computation for channel with initial T = {T}")

    while True:
        # Group pixels based on the current threshold
        G1 = channel[channel <= T]
        G2 = channel[channel > T]

        # Check if groups are empty and handle this edge case
        if G1.size == 0 or G2.size == 0:
            print("One of the groups is empty, returning initial threshold.")
            return T

        # Compute mean intensity of both groups
        mu1 = np.mean(G1)
        mu2 = np.mean(G2)

        # Update the threshold
        T_new = (mu1 + mu2) / 2

        # Debugging: Print the new threshold value for each iteration
        print(f"Updated threshold: T_new = {T_new}")

        # Check for convergence
        if abs(T - T_new) < epsilon:
            break

        T = T_new  # Update threshold for next iteration

    print(f"Final threshold for channel: {T}")
    return T


def plot_histograms(image):
    """
    Plot histograms of each color channel to understand the pixel distribution.
    """
    colors = ['Red', 'Green', 'Blue']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        channel = image[:, :, i]
        axes[i].hist(channel.flatten(), bins=256, color=colors[i].lower(), alpha=0.7)
        axes[i].set_title(f'{colors[i]} Channel Histogram')

    plt.show()


def automatic(image, initial_threshold=128, epsilon=1):
    """
    Perform automatic thresholding on an image.
    Works for both grayscale and RGB images.
    """
    thresholds = []
    if len(image.shape) == 2:  # Grayscale image
        threshold = compute_threshold(image, initial_threshold, epsilon)
        thresholds.append(threshold)
        binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)
        return binary_image

    else:  # RGB image
        binary_image = np.zeros_like(image, dtype=np.uint8)

        # Apply threshold independently to each channel
        for i in range(3):
            channel = image[:, :, i]
            threshold = compute_threshold(channel, initial_threshold, epsilon)
            thresholds.append(threshold)
            binary_image[:, :, i] = np.where(channel > threshold, 255, 0)

        return binary_image.astype(np.float32) / 255.0, thresholds


# Example usage
image_path = 'images/baboon.png'  # Replace with your actual image path
image = plt.imread(image_path).astype(np.float32)

# Display histograms of each channel to understand pixel distribution
plot_histograms(image)

# Apply the color-preserving threshold
binary_image, thresholds = automatic(image)

# Display thresholds for each channel
print("Thresholds for each channel:", thresholds)

# Plot results for verification
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image.astype(np.uint8))
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(binary_image)
ax[1].set_title(f'Color Thresholded Image\n(R={thresholds[0]}, G={thresholds[1]}, B={thresholds[2]})')
ax[1].axis('off')

plt.show()