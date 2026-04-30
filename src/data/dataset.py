import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class RectangleDataset:
    """Custom dataset class for rectangle detection without PyTorch dependency.
    
    This dataset loads images and corresponding YOLO-format labels for rectangle detection.
    Images are automatically normalized and converted to CHW format for model compatibility.
    
    Attributes:
        image_dir: Path to directory containing image files
        label_dir: Path to directory containing label files
        image_files: Sorted list of image filenames
        normalize: Whether to normalize images to [-1, 1] range
    """
    
    def __init__(self, data_dir, normalize=True):
        print(f"📂 Initializing dataset from: {data_dir}")
        self.image_dir = os.path.join(data_dir, "images")
        self.label_dir = os.path.join(data_dir, "labels")
        
        print("🔍 Scanning for image files...")
        all_files = os.listdir(self.image_dir)
        self.image_files = sorted([f for f in all_files if f.endswith(('.jpg', '.png'))])
        print(f"✅ Found {len(self.image_files)} image files")
        
        self.normalize = normalize
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image, label) where image is normalized CHW format and label is YOLO format
        """
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Load label
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'r') as f:
            label = np.array([float(x) for x in f.read().strip().split()])
        
        # Convert image to float and normalize if needed
        image = image.astype(np.float32) / 255.0
        if self.normalize:
            image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Convert from HWC to CHW format for consistency
        image = image.transpose(2, 0, 1)
        
        return image, label
    
    def get_batch(self, indices):
        """Get a batch of samples efficiently.
        
        Args:
            indices: List of indices to retrieve from the dataset
            
        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        images = []
        labels = []
        
        for idx in indices:
            image, label = self[idx]
            images.append(image)
            labels.append(label)
        
        return np.array(images), np.array(labels)


def get_dataloaders(data_dir, batch_size=32, val_split=0.2, random_state=42, num_workers=0):
    """Create train and validation dataloaders using scikit-learn for splitting.
    
    Args:
        data_dir: Directory containing the dataset with 'images' and 'labels' subdirectories
        batch_size: Number of samples per batch
        val_split: Fraction of data to use for validation
        random_state: Random seed for reproducible splitting
        num_workers: Unused parameter kept for API compatibility
        
    Returns:
        Tuple of (train_loader, val_loader) dataloaders
    """
    
    # Create full dataset
    full_dataset = RectangleDataset(data_dir)
    
    # Get all indices
    indices = np.arange(len(full_dataset))
    
    # Split indices using scikit-learn
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=val_split, 
        random_state=random_state
    )
    
    # Create dataloaders with direct indexing
    class SimpleDataLoader:
        def __init__(self, dataset, indices, batch_size, shuffle=False):
            self.dataset = dataset
            self.indices = indices
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.current_idx = 0
            
        def __iter__(self):
            if self.shuffle:
                indices = np.random.permutation(self.indices)
            else:
                indices = self.indices.copy()
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                images = []
                labels = []
                for idx in batch_indices:
                    img, label = self.dataset[idx]
                    images.append(img)
                    labels.append(label)
                yield np.array(images), np.array(labels)
        
        def __len__(self):
            return (len(self.indices) + self.batch_size - 1) // self.batch_size
    
    train_loader = SimpleDataLoader(full_dataset, train_indices, batch_size, shuffle=True)
    val_loader = SimpleDataLoader(full_dataset, val_indices, batch_size, shuffle=False)
    
    print(f"Dataset split complete:")
    print(f"  - Total samples: {len(full_dataset)}")
    print(f"  - Training samples: {len(train_indices)}")
    print(f"  - Validation samples: {len(val_indices)}")
    
    return train_loader, val_loader
