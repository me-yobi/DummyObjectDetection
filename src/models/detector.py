import numpy as np

class SimpleRectangleDetector:
    """
    Simple rectangle detector using direct regression from convolution features.
    No deep learning - just direct computation of bounding box from edge features.
    """
    
    def __init__(self, image_size=256):
        self.image_size = image_size
        self._init_kernels()
        
    def _init_kernels(self):
        """Initialize convolution kernels for edge detection"""
        # Sobel operators for edge detection
        self.h_kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)  # Horizontal edges (Sobel X)
        
        self.v_kernel = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)  # Vertical edges (Sobel Y)
        
        self.c_kernel = np.array([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)  # Corner detector (Laplacian)
    
    def conv2d(self, image, kernel):
        """2D convolution with valid padding"""
        # Handle both 2D and 3D images
        if len(image.shape) == 2:
            h, w = image.shape
            c = 1
            image_2d = image
        else:
            h, w, c = image.shape
            image_2d = np.mean(image, axis=2)
        
        # Handle both 2D and 3D kernels
        if len(kernel.shape) == 2:
            kh, kw = kernel.shape
            kernel_2d = kernel
        else:
            kh, kw, kc = kernel.shape
            kernel_2d = np.mean(kernel, axis=2)
        
        # Output dimensions (valid padding)
        out_h = h - kh + 1
        out_w = w - kw + 1
        
        output = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                output[i, j] = np.sum(image_2d[i:i+kh, j:j+kw] * kernel_2d)
        return output
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def maxpool2d(self, feature_map, size=2):
        """2D max pooling"""
        h, w = feature_map.shape
        pooled_h = h // size
        pooled_w = w // size
        
        output = np.zeros((pooled_h, pooled_w))
        for i in range(pooled_h):
            for j in range(pooled_w):
                output[i, j] = np.max(feature_map[i*size:(i+1)*size, j*size:(j+1)*size])
        
        return output
    
    def find_edges(self, features, threshold=0.1):
        """Find edge positions from feature map using custom contour detection"""
        # Threshold the features
        binary = (features > threshold).astype(np.uint8)
        
        # Find contours using custom algorithm
        contours = self._find_contours_custom(binary)
        
        return contours
    
    def find_contours(self, image):
        """Find contours in binary image using custom algorithm"""
        return self._find_contours_custom(image)
    
    def contour_area(self, contour):
        """Compute area of contour using shoelace formula"""
        if len(contour) < 3:
            return 0.0
        
        # Reshape contour to get (n, 2) array of points
        points = contour.reshape(-1, 2).astype(np.float64)
        
        # Apply shoelace formula
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        
        return float(area)
    
    def bounding_rect(self, contour):
        """Compute bounding rectangle of contour"""
        if len(contour) == 0:
            return (0, 0, 0, 0)
        
        # Reshape contour to get (n, 2) array of points
        points = contour.reshape(-1, 2)
        
        # Find min and max coordinates
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        
        # Convert to int and calculate width/height
        x = int(min_x)
        y = int(min_y)
        w = int(max_x - min_x)
        h = int(max_y - min_y)
        
        return (x, y, w, h)
    
    def _find_contours_custom(self, binary_image):
        """
        Custom contour detection algorithm using numpy.
        Implements a boundary following algorithm to find contours in binary images.
        """
        # Ensure binary image is uint8 with values 0 or 255
        binary = binary_image.copy()
        if binary.max() <= 1:
            binary = (binary * 255).astype(np.uint8)
        else:
            binary = binary.astype(np.uint8)
        
        h, w = binary.shape
        visited = np.zeros((h, w), dtype=bool)
        contours = []
        
        # Find all boundary pixels (pixels that have at least one background neighbor)
        for i in range(1, h-1):
            for j in range(1, w-1):
                if binary[i, j] > 0 and not visited[i, j]:
                    # Check if this is a boundary pixel
                    is_boundary = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                if binary[ni, nj] == 0:
                                    is_boundary = True
                                    break
                        if is_boundary:
                            break
                    
                    if is_boundary:
                        # Trace contour from this boundary pixel
                        contour = self._trace_contour(binary, visited, i, j)
                        if len(contour) > 2:  # Only keep contours with more than 2 points
                            contours.append(np.array(contour, dtype=np.int32).reshape(-1, 1, 2))
        
        return contours
    
    def _trace_contour(self, binary, visited, start_i, start_j):
        """
        Trace a single contour starting from (start_i, start_j) using boundary following.
        """
        h, w = binary.shape
        contour = []
        
        # 8-connected neighborhood directions (clockwise)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        
        current_i, current_j = start_i, start_j
        start_dir = 0  # Start searching from direction 0
        
        while True:
            # Mark current pixel as visited
            visited[current_i, current_j] = True
            contour.append([current_j, current_i])  # Store as (x, y)
            
            # Find next boundary pixel
            found_next = False
            for dir_offset in range(8):
                dir_idx = (start_dir + dir_offset) % 8
                di, dj = directions[dir_idx]
                next_i, next_j = current_i + di, current_j + dj
                
                if (0 <= next_i < h and 0 <= next_j < w and 
                    binary[next_i, next_j] > 0 and not visited[next_i, next_j]):
                    
                    # Check if this is a boundary pixel
                    is_boundary = False
                    for ddi in [-1, 0, 1]:
                        for ddj in [-1, 0, 1]:
                            if ddi == 0 and ddj == 0:
                                continue
                            ni, nj = next_i + ddi, next_j + ddj
                            if 0 <= ni < h and 0 <= nj < w:
                                if binary[ni, nj] == 0:
                                    is_boundary = True
                                    break
                        if is_boundary:
                            break
                    
                    if is_boundary:
                        current_i, current_j = next_i, next_j
                        start_dir = (dir_idx + 6) % 8  # Start searching from previous direction
                        found_next = True
                        break
            
            # If we can't find next pixel or we're back to start, finish
            if not found_next or (current_i == start_i and current_j == start_j):
                break
        
        return contour
    
    def direct_regression(self, image):
        """
        Directly compute bounding box using custom convolution edge detection.
        Finds the rectangle border in the image.
        """
        # Convert from [-1, 1] to [0, 1] if needed
        if image.min() < 0:
            image = (image + 1.0) / 2.0
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()
        
        # Custom edge detection using convolution kernels directly on grayscale image
        # Horizontal edges
        h_edges = self.conv2d(gray, self.h_kernel)
        # Vertical edges  
        v_edges = self.conv2d(gray, self.v_kernel)
        # Combined edge strength
        edges = np.sqrt(h_edges**2 + v_edges**2)
        
        # Apply threshold to create binary edge map
        # Use adaptive threshold based on edge strength statistics
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        threshold = edge_mean + 2 * edge_std  # 2 sigma above mean
        
        # Create binary edge map
        edge_map = (edges > threshold).astype(np.uint8) * 255
        
        # Custom morphological operations using convolution
        # Dilation kernel (3x3)
        dilate_kernel = np.ones((3, 3), dtype=np.float32)
        dilated = self.conv2d(edge_map.astype(np.float32), dilate_kernel)
        dilated = (dilated > 0).astype(np.uint8) * 255
        
        # Find contours using custom approach
        contours = self.find_contours(dilated)
        
        if contours:
            # Find the largest contour that's not the entire image
            h_img, w_img = image.shape[:2]
            image_area = h_img * w_img
            
            # Filter out contours that are too large (likely the image border)
            valid_contours = []
            for contour in contours:
                area = self.contour_area(contour)
                if area < image_area * 0.5:  # Must be less than half the image area
                    valid_contours.append(contour)
            
            if valid_contours:
                # Find the largest valid contour
                largest_contour = max(valid_contours, key=self.contour_area)
                area = self.contour_area(largest_contour)
                
                # Filter out very small areas
                if area > 500:  # Minimum area threshold
                    # Get bounding rectangle
                    x, y, w, h = self.bounding_rect(largest_contour)
                    
                    # Convert to normalized coordinates [0, 1]
                    x_center = (x + w / 2) / w_img
                    y_center = (y + h / 2) / h_img
                    width = w / w_img
                    height = h / h_img
                    
                    # Ensure values are in [0, 1]
                    x_center = np.clip(x_center, 0, 1)
                    y_center = np.clip(y_center, 0, 1)
                    width = np.clip(width, 0, 1)
                    height = np.clip(height, 0, 1)
                    
                    return np.array([1.0, x_center, y_center, width, height])

        # No object detected using edge-based method
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    def forward(self, image):
        """Forward pass - just direct regression"""
        return self.direct_regression(image)
    
    def train(self):
        """No-op - no training needed"""
        pass
    
    def eval(self):
        """No-op - no training mode"""
        pass
    
    def train_step(self, image, target, learning_rate=0.001):
        """No-op - no training needed"""
        # Just return the prediction
        output = self.forward(image)
        loss = np.mean((output - target) ** 2)
        return loss, output


def create_model(device):
    """Create model instance"""
    return SimpleRectangleDetector()
