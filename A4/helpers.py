import numpy as np

def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    result = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)
    return result

def draw_line(image, point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        image[y1, x1] = 255
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return image


def draw_triangle(image, p1, p2, p3):
    def edge_function(v0, v1, p):
        return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])

    if len(image.shape) > 2:
        image = image[:, :, 0]

    height, width = image.shape
    bounding_box_min = (
        max(0, min(p1[0], p2[0], p3[0])),
        max(0, min(p1[1], p2[1], p3[1])),
    )
    bounding_box_max = (
        min(width - 1, max(p1[0], p2[0], p3[0])),
        min(height - 1, max(p1[1], p2[1], p3[1])),
    )

    for y in range(bounding_box_min[1], bounding_box_max[1] + 1):
        for x in range(bounding_box_min[0], bounding_box_max[0] + 1):
            p = (x, y)
            w0 = edge_function(p2, p3, p)
            w1 = edge_function(p3, p1, p)
            w2 = edge_function(p1, p2, p)

            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                image[y, x] = 255
    return image


def draw_circle(image, center, radius):
    cx, cy = center
    x, y = 0, radius
    d = 1 - radius
    while x <= y:
        for dx, dy in [(x, y), (y, x), (-x, y), (-y, x), (-x, -y), (-y, -x), (x, -y), (y, -x)]:
            if 0 <= cx + dx < image.shape[1] and 0 <= cy + dy < image.shape[0]:
                image[cy + dy, cx + dx] = 255
        if d < 0:
            d += 2 * x + 3
        else:
            d += 2 * (x - y) + 5
            y -= 1
        x += 1
    return image

def convert_to_grayscale(image):
    if len(image.shape) == 3:  # Color image
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image  # Already grayscale


def sobel_edge_detection(image):
    # Ensure the image is grayscale
    if len(image.shape) == 3:  # Convert RGB to grayscale if necessary
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        image = image.astype(np.uint8)  # Ensure integer type

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    grad_x = convolve(image, kernel_x)
    grad_y = convolve(image, kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = (magnitude / magnitude.max()) * 255  # Normalize
    return magnitude.astype(np.uint8)

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

def clamp(x, lower, upper):
    return max(lower, min(x, upper))

def gaussian_blur(image, kernel_size=5, sigma=1.4):
    # Create Gaussian kernel
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = np.outer(gauss, gauss)
    kernel /= kernel.sum()
    # Convolve with the image
    return convolve(image, kernel)

def compute_gradients(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * (180.0 / np.pi)  # Convert to degrees
    direction = (direction + 180) % 180  # Normalize to [0, 180]
    return magnitude, direction

def non_maximum_suppression(magnitude, direction):
    suppressed = np.zeros_like(magnitude)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = direction[i, j]
            q = 255
            r = 255

            # Check direction
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed

def double_threshold(image, low_threshold, high_threshold):
    """Apply double thresholding."""
    strong = 255
    weak = 75

    result = np.zeros_like(image)
    strong_row, strong_col = np.where(image >= high_threshold)
    weak_row, weak_col = np.where((image >= low_threshold) & (image < high_threshold))

    result[strong_row, strong_col] = strong
    result[weak_row, weak_col] = weak

    return result, strong, weak

def edge_tracking(image, strong, weak):
    """Track edges by hysteresis."""
    rows, cols = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if image[i, j] == weak:
                if (image[i + 1, j - 1] == strong or image[i + 1, j] == strong or image[i + 1, j + 1] == strong
                        or image[i, j - 1] == strong or image[i, j + 1] == strong
                        or image[i - 1, j - 1] == strong or image[i - 1, j] == strong or image[i - 1, j + 1] == strong):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

def canny_edge_detection(image, low_threshold=50, high_threshold=125):
    # Ensure grayscale
    if len(image.shape) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    # Gaussian smoothing
    blurred = gaussian_blur(image)

    # Gradient calculation
    magnitude, direction = compute_gradients(blurred)

    # Non-maximum suppression
    thin_edges = non_maximum_suppression(magnitude, direction)

    # Double thresholding
    thresholded, strong, weak = double_threshold(thin_edges, low_threshold, high_threshold)

    # Edge tracking by hysteresis
    final_edges = edge_tracking(thresholded, strong, weak)

    return final_edges

