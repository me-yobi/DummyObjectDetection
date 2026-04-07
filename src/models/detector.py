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
                        # This is the exact same code as in FIND contour... uses next i and next j
                    
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
        
        # Apply corner detection using Laplacian kernel
        corner_response = self.detect_corners(gray)
        
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
                # Score contours by corner validation
                scored_contours = []
                for contour in valid_contours:
                    # Get bounding box
                    x, y, w, h = self.bounding_rect(contour)
                    
                    # Validate corners
                    corner_score = self.validate_corners(corner_response, x, y, w, h)
                    area = self.contour_area(contour)
                    
                    # Combined score: area * corner_quality
                    scored_contours.append((contour, corner_score, area, x, y, w, h))
                
                # Sort by corner score (descending)
                scored_contours.sort(key=lambda x: x[1], reverse=True)
                
                # Select best contour with sufficient corner validation
                best_contour = None
                best_x, best_y, best_w, best_h = 0, 0, 0, 0
                
                for contour, corner_score, area, x, y, w, h in scored_contours:
                    if area > 500:  # Minimum area threshold
                        if corner_score > 0.3:  # Require at least 30% corner quality
                            best_contour = contour
                            best_x, best_y, best_w, best_h = x, y, w, h
                            break
                
                # Fallback to largest if no good corners found
                if best_contour is None and scored_contours:
                    for contour, corner_score, area, x, y, w, h in scored_contours:
                        if area > 500:
                            best_contour = contour
                            best_x, best_y, best_w, best_h = x, y, w, h
                            break
                
                if best_contour is not None:
                    # Refine bounding box using corner detection
                    refined_x, refined_y, refined_w, refined_h = self.refine_box_with_corners(
                        corner_response, best_x, best_y, best_w, best_h
                    )
                    
                    # Convert to normalized coordinates [0, 1]
                    x_center = (refined_x + refined_w / 2) / w_img
                    y_center = (refined_y + refined_h / 2) / h_img
                    width = refined_w / w_img
                    height = refined_h / h_img
                    
                    # Ensure values are in [0, 1]
                    x_center = np.clip(x_center, 0, 1)
                    y_center = np.clip(y_center, 0, 1)
                    width = np.clip(width, 0, 1)
                    height = np.clip(height, 0, 1)
                    
                    return np.array([1.0, x_center, y_center, width, height])

        # No object detected using edge-based method
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    def detect_corners(self, gray_image):
        """
        Detect corners using Laplacian kernel.
        Returns corner response map (absolute values, high = strong corner).
        """
        # Apply Laplacian kernel
        corners = self.conv2d(gray_image, self.c_kernel)
        # Take absolute value (corners produce strong positive or negative response)
        return np.abs(corners)
    
    def validate_corners(self, corner_response, x, y, w, h, margin=10):
        """
        Validate that a bounding box has strong corner responses at its corners.
        Returns a score between 0 and 1 indicating corner quality.
        """
        h_map, w_map = corner_response.shape
        
        # Define corner regions (with margin)
        corners = [
            (max(0, x - margin), max(0, y - margin), x + margin, y + margin),  # Top-left
            (min(w_map, x + w - margin), max(0, y - margin), min(w_map, x + w + margin), y + margin),  # Top-right
            (max(0, x - margin), min(h_map, y + h - margin), x + margin, min(h_map, y + h + margin)),  # Bottom-left
            (min(w_map, x + w - margin), min(h_map, y + h - margin), min(w_map, x + w + margin), min(h_map, y + h + margin))  # Bottom-right
        ]
        
        corner_scores = []
        for x1, y1, x2, y2 in corners:
            if x2 > x1 and y2 > y1:
                # Extract corner region
                region = corner_response[y1:y2, x1:x2]
                if region.size > 0:
                    # Score is max response in region, normalized
                    max_response = np.max(region)
                    # Normalize by global max (add small epsilon to avoid div by zero)
                    global_max = np.max(corner_response) + 1e-8
                    corner_scores.append(max_response / global_max)
        
        # Return average corner score (0 to 1)
        return np.mean(corner_scores) if corner_scores else 0.0
    
    def refine_box_with_corners(self, corner_response, x, y, w, h, search_margin=15):
        """
        Refine bounding box by aligning with detected corners.
        Searches for strongest corner responses near the estimated corners.
        """
        h_map, w_map = corner_response.shape
        
        # Define search regions for each corner
        search_regions = [
            # (start_x, start_y, end_x, end_y, corner_name)
            (max(0, x - search_margin), max(0, y - search_margin), 
             min(w_map, x + search_margin), min(h_map, y + search_margin), 'tl'),
            (max(0, x + w - search_margin), max(0, y - search_margin),
             min(w_map, x + w + search_margin), min(h_map, y + search_margin), 'tr'),
            (max(0, x - search_margin), max(0, y + h - search_margin),
             min(w_map, x + search_margin), min(h_map, y + h + search_margin), 'bl'),
            (max(0, x + w - search_margin), max(0, y + h - search_margin),
             min(w_map, x + w + search_margin), min(h_map, y + h + search_margin), 'br')
        ]
        
        refined_corners = []
        
        for sx1, sy1, sx2, sy2, name in search_regions:
            if sx2 > sx1 and sy2 > sy1:
                region = corner_response[sy1:sy2, sx1:sx2]
                if region.size > 0:
                    # Find position of maximum response
                    max_idx = np.unravel_index(np.argmax(region), region.shape)
                    # Convert to global coordinates
                    refined_y = sy1 + max_idx[0]
                    refined_x = sx1 + max_idx[1]
                    refined_corners.append((refined_x, refined_y, np.max(region)))
                else:
                    # Fallback to original corner
                    if name == 'tl':
                        refined_corners.append((x, y, 0))
                    elif name == 'tr':
                        refined_corners.append((x + w, y, 0))
                    elif name == 'bl':
                        refined_corners.append((x, y + h, 0))
                    else:  # br
                        refined_corners.append((x + w, y + h, 0))
        
        if len(refined_corners) == 4:
            # Calculate refined box from corners
            # Use average of top corners for left edge, bottom for right, etc.
            tl_x, tl_y, _ = refined_corners[0]
            tr_x, tr_y, _ = refined_corners[1]
            bl_x, bl_y, _ = refined_corners[2]
            br_x, br_y, _ = refined_corners[3]
            
            # Average to reduce noise
            left_x = int((tl_x + bl_x) / 2)
            right_x = int((tr_x + br_x) / 2)
            top_y = int((tl_y + tr_y) / 2)
            bottom_y = int((bl_y + br_y) / 2)
            
            refined_x = max(0, left_x)
            refined_y = max(0, top_y)
            refined_w = max(10, right_x - left_x)  # Ensure minimum size
            refined_h = max(10, bottom_y - top_y)
            
            return refined_x, refined_y, refined_w, refined_h
        
        # Fallback to original if refinement failed
        return x, y, w, h
    
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
