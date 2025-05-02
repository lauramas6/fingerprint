import numpy as np

def extract_feature_vector(minutiae, img_shape, grid_size=8):
    """Generate grid histogram feature vector from minutiae points."""
    h, w = img_shape[:2]
    grid = np.zeros((grid_size, grid_size))

    for (y, x) in minutiae:
        row = min(grid_size-1, int(y / (h / grid_size)))
        col = min(grid_size-1, int(x / (w / grid_size)))
        grid[row, col] += 1

    return grid.flatten()
