import numpy as np

# Helper function to clamp indices to the image boundaries
def clamp(x, lower, upper):
    return max(lower, min(x, upper))

def kuwahara_filter(image, window_size=5):
    half_size = window_size // 2
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)  # Add a channel dimension

    height, width, channels = image.shape
    padded_image = np.pad(image, ((half_size, half_size), (half_size, half_size), (0, 0)), mode='edge')
    filtered_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            for c in range(channels):
                region1 = padded_image[i:i + half_size + 1, j:j + half_size + 1, c]
                region2 = padded_image[i:i + half_size + 1, j + 1:j + half_size + 2, c]
                region3 = padded_image[i + 1:i + half_size + 2, j:j + half_size + 1, c]
                region4 = padded_image[i + 1:i + half_size + 2, j + 1:j + half_size + 2, c]
                regions = [region1, region2, region3, region4]
                variances = [np.var(r) for r in regions]
                means = [np.mean(r) for r in regions]
                filtered_image[i, j, c] = means[np.argmin(variances)]

    # Remove the extra channel dimension for grayscale images
    if filtered_image.shape[2] == 1:
        filtered_image = np.squeeze(filtered_image)

    return filtered_image

def triangle_filter(image):
    kernel= np.array([1, 2, 3, 2, 1], dtype=float)

    # Normalize kernel
    kernel /= np.sum(kernel)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    height, width, channels = image.shape
    result = np.zeros_like(image)

    # Horizontal pass
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                start_col = clamp(j - 2, 0, width - 1)
                end_col = clamp(j + 3, 0, width)
                slice_ = image[i, start_col:end_col, c]

                #handle edge padding
                if slice_.shape[0] < 5:
                    slice_ = np.pad(slice_, (0, 5 - slice_.shape[0]), mode='edge')
                result[i, j, c] = np.sum(slice_ * kernel)

    # Vertical pass
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                start_row = clamp(i - 2, 0, height - 1)
                end_row = clamp(i + 3, 0, height)
                slice_ = result[start_row:end_row, j, c]

                if slice_.shape[0] < 5:
                    slice_ = np.pad(slice_, (0, 5 - slice_.shape[0]), mode='edge')

                result[i, j, c] = np.sum(slice_ * kernel)

    if result.shape[2] == 1:
        result = np.squeeze(result)

    return result

def median_filter(image):
    kernel_size = 5
    half_size = kernel_size // 2
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    height, width, channels = image.shape
    result = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            for c in range(channels):
                neighborhood = image[
                    clamp(i - half_size, 0, height - 1):clamp(i + half_size + 1, 0, height - 1),
                    clamp(j - half_size, 0, width - 1):clamp(j + half_size + 1, 0, width - 1),
                    c]
                if neighborhood.size < kernel_size * kernel_size:
                    neighborhood = np.pad(neighborhood, ((half_size, half_size), (half_size, half_size)), mode='edge')

                result[i, j, c] = np.median(neighborhood)

    if result.shape[2] == 1:
        result = np.squeeze(result)

    return result

def gaussian_kernel(size, sigma=1.0):
    kernel_size = size // 2
    x = np.arange(-kernel_size, kernel_size + 1)
    gaussian_1d = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    gaussian_1d /= gaussian_1d.sum()
    return gaussian_1d

def gaussian_filter(image, sigma, kernel_size=5):
    kernel_1d = gaussian_kernel(kernel_size, sigma)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    height, width, channels = image.shape
    filtered_image = np.zeros_like(image, dtype=float)

    for c in range(channels):
        # Horizontal pass
        for i in range(height):
            for j in range(width):
                start_col = clamp(j - 2, 0, width - 1)
                end_col = clamp(j + 3, 0, width)
                slice_ = image[i, start_col:end_col, c]
                if slice_.shape[0] < kernel_size:
                    slice_ = np.pad(slice_, (0, kernel_size - slice_.shape[0]), mode='edge')

                filtered_image[i, j, c] = np.sum(slice_ * kernel_1d)

        # Vertical pass
        for j in range(width):
            for i in range(height):
                start_row = clamp(i - 2, 0, height - 1)
                end_row = clamp(i + 3, 0, height)
                slice_ = filtered_image[start_row:end_row, j, c]
                if slice_.shape[0] < kernel_size:
                    slice_ = np.pad(slice_, (0, kernel_size - slice_.shape[0]), mode='edge')
                filtered_image[i, j, c] = np.sum(slice_ * kernel_1d)

    # Normalize the filtered image to stay within the valid range
    if image.max() <= 1.0:
        filtered_image = np.clip(filtered_image, 0, 1)
    else:
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    if filtered_image.shape[2] == 1:
        filtered_image = np.squeeze(filtered_image)

    return filtered_image

def mean_filter(image, kernel_size=5):
    half_size = kernel_size // 2
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    height, width, channels = image.shape
    filtered_image = np.zeros((height, width, channels), dtype=float)
    mean_kernel = np.ones(kernel_size) / kernel_size

    # Horizontal pass
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                start_col = clamp(j - half_size, 0, width - 1)
                end_col = clamp(j + half_size + 1, 0, width)
                neighborhood = image[i, start_col:end_col, c]
                if neighborhood.shape[0] < kernel_size:
                    neighborhood = np.pad(neighborhood, (0, kernel_size - neighborhood.shape[0]), mode='edge')
                filtered_image[i, j, c] = np.sum(neighborhood * mean_kernel)

    # Vertical pass
    for c in range(channels):
        for j in range(width):
            for i in range(height):
                start_row = clamp(i - half_size, 0, height - 1)
                end_row = clamp(i + half_size + 1, 0, height)
                neighborhood = filtered_image[start_row:end_row, j, c]
                if neighborhood.shape[0] < kernel_size:
                    neighborhood = np.pad(neighborhood, (0, kernel_size - neighborhood.shape[0]), mode='edge')
                filtered_image[i, j, c] = np.sum(neighborhood * mean_kernel)

    if filtered_image.max() > 255:
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    else:
        filtered_image = np.clip(filtered_image, 0, 1).astype(np.float32)

    if filtered_image.shape[2] == 1:
        filtered_image = np.squeeze(filtered_image)
    return filtered_image