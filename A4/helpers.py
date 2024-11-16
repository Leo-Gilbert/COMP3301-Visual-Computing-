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
