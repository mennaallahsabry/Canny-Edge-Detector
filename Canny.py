import numpy as np
import cv2

def conv(image, kernel):
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    for i in range(Hi):
        for j in range(Wi):
            window = padded[i:i+Hk, j:j+Wk]
            out[i, j] = np.sum(window * kernel)

    return out

#(1 / (2 * π * σ^2)) * exp(-((x - size//2)^2 + (y - size//2)^2) / (2 * σ^2))
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def partial_x(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return conv(img, sobel_x)

def partial_y(img):
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return conv(img, sobel_y)

#(∂f/∂x)s the partial derivative of x.
#(∂f/∂y)s the partial derivative of y.
#gradient magnitude= "√( (∂f/∂x)^2 + (∂f/∂y)^2 )"
#gradient direction= arctan(∂x/∂f ∂y/∂f) and multiple with (180 / np.pi) to convert the angle from radians to degrees.
def gradient(img):
    Gx = partial_x(img)
    Gy = partial_y(img)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    theta = np.arctan2(Gy, Gx) * (180 / np.pi)
    return G, theta

#examines each pixel's gradient magnitude and compares it to its neighbors along the gradient direction.
def non_maximum_suppression(G, theta):
    H, W = G.shape
    out = np.zeros((H, W))

    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            direction = theta[i, j]
            if direction == 0:
                neighbors = [(i, j - 1), (i, j + 1)]
            elif direction == 45:
                neighbors = [(i - 1, j + 1), (i + 1, j - 1)]
            elif direction == 90:
                neighbors = [(i - 1, j), (i + 1, j)]
            else:
                neighbors = [(i - 1, j - 1), (i + 1, j + 1)]
            values = [G[n[0], n[1]] for n in neighbors]
            if G[i, j] >= max(values):
                out[i, j] = G[i, j]

    return out
#determine the edges base on being higher that thresholding or between two strong edges 
def double_thresholding(img, high, low):
    strong_edges = img > high
    weak_edges = (img >= low) & (img <= high)
    return strong_edges, weak_edges


#designed to return a list of neighboring coordinates around a given point (y, x) in a 2D grid
#(i == y and j == x): This part ensures that the center point itself is not included in the list of neighbors.
#i < 0 or i >= H or j < 0 or j >= W: These parts make sure that coordinates outside the grid boundaries are not included.
def get_neighbors(y, x, H, W):
    neighbors = []
    for i in range(y - 1, y + 2):
        for j in range(x - 1, x + 2):
            if (i == y and j == x) or i < 0 or i >= H or j < 0 or j >= W:
                continue
            neighbors.append((i, j))
    return neighbors
# part of edge linking or edge tracking
def link_edges(strong_edges, weak_edges):
    H, W = strong_edges.shape
    edges = np.copy(strong_edges)
    stack = []

    for i in range(H):
        for j in range(W):
            if strong_edges[i, j]:
                stack.append((i, j))
                while stack:
                    y, x = stack.pop()
                    neighbors = get_neighbors(y, x, H, W)
                    for n in neighbors:
                        ny, nx = n
                        if weak_edges[ny, nx] and not edges[ny, nx]:
                            edges[ny, nx] = True
                            stack.append(n)

    return edges
# connect all functions to implement canny algorithm 
def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_img = conv(img, kernel)
    G, theta = gradient(blurred_img)
    nms_img = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms_img, high, low)
    edges = link_edges(strong_edges, weak_edges)
    return edges

# Load an image in grayscale
image = cv2.imread('Coral_colony_photo_1_year_ago.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = canny(image)

import matplotlib.pyplot as plt

# Display the result using matplotlib
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.show()

