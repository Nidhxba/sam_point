# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import json
# import os
# from sklearn.cluster import DBSCAN
# import rasterio
# from rasterio.windows import Window
# from rasterio.enums import Resampling
# import tifffile
# from PIL import Image
# from shapely.geometry import Point, Polygon

# class RGBTifTreeCounter:
#     """Simplified tree counter optimized for RGB TIF files"""
    
#     def __init__(self):
#         self.image = None
#         self.metadata = {}
#         self.segments = []
#         self.tree_centers = []
        
#     def load_rgb_tif(self, tif_path, max_size=2048):
#         """
#         Load RGB TIF file with automatic resizing if needed
        
#         Args:
#             tif_path: Path to RGB TIF file
#             max_size: Maximum dimension to resize to if image is larger
#         """
#         print(f"Loading RGB TIF: {tif_path}")
        
#         try:
#             # Try rasterio first for geospatial TIFs
#             with rasterio.open(tif_path) as src:
#                 self.metadata = {
#                     'crs': src.crs,
#                     'transform': src.transform,
#                     'bounds': src.bounds,
#                     'width': src.width,
#                     'height': src.height,
#                     'count': src.count,
#                     'dtype': str(src.dtypes[0])
#                 }
                
#                 print(f"TIF info: {src.count} bands, {src.width}x{src.height}, {src.dtypes[0]}")
                
#                 # Check if we need to resize
#                 if src.width > max_size or src.height > max_size:
#                     scale = max_size / max(src.width, src.height)
#                     new_width = int(src.width * scale)
#                     new_height = int(src.height * scale)
                    
#                     print(f"Resizing from {src.width}x{src.height} to {new_width}x{new_height}")
                    
#                     # Read RGB bands with resampling
#                     rgb_bands = [1, 2, 3] if src.count >= 3 else [1]
#                     self.image = src.read(
#                         rgb_bands,
#                         out_shape=(len(rgb_bands), new_height, new_width),
#                         resampling=Resampling.bilinear
#                     )
#                     self.image = np.transpose(self.image, (1, 2, 0))
#                 else:
#                     # Read RGB bands at full resolution
#                     rgb_bands = [1, 2, 3] if src.count >= 3 else [1]
#                     self.image = src.read(rgb_bands)
#                     self.image = np.transpose(self.image, (1, 2, 0))
                
#                 # Handle single band (convert to RGB)
#                 if len(self.image.shape) == 2 or self.image.shape[2] == 1:
#                     if len(self.image.shape) == 2:
#                         self.image = np.stack([self.image, self.image, self.image], axis=-1)
#                     else:
#                         self.image = np.repeat(self.image, 3, axis=2)
        
#         except Exception as e:
#             print(f"Rasterio failed: {e}. Trying alternative methods...")
            
#             # Fallback to tifffile
#             try:
#                 self.image = tifffile.imread(tif_path)
                
#                 if len(self.image.shape) == 3 and self.image.shape[0] < self.image.shape[2]:
#                     self.image = np.transpose(self.image, (1, 2, 0))
                
#                 # Take first 3 channels
#                 if len(self.image.shape) == 3 and self.image.shape[2] > 3:
#                     self.image = self.image[:, :, :3]
#                 elif len(self.image.shape) == 2:
#                     self.image = np.stack([self.image, self.image, self.image], axis=-1)
                
#                 # Resize if needed
#                 if max(self.image.shape[:2]) > max_size:
#                     h, w = self.image.shape[:2]
#                     scale = max_size / max(h, w)
#                     new_h, new_w = int(h * scale), int(w * scale)
#                     self.image = cv2.resize(self.image, (new_w, new_h))
                    
#             except Exception as e2:
#                 print(f"Tifffile failed: {e2}. Trying PIL...")
                
#                 # Final fallback
#                 with Image.open(tif_path) as img:
#                     # Resize if needed
#                     if max(img.size) > max_size:
#                         scale = max_size / max(img.size)
#                         new_size = (int(img.width * scale), int(img.height * scale))
#                         img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
#                     self.image = np.array(img)
                    
#                     if len(self.image.shape) == 2:
#                         self.image = np.stack([self.image, self.image, self.image], axis=-1)
        
#         # Normalize to uint8
#         if self.image.dtype != np.uint8:
#             if self.image.dtype == np.uint16:
#                 self.image = (self.image / 256).astype(np.uint8)
#             elif self.image.dtype in [np.float32, np.float64]:
#                 if self.image.max() <= 1.0:
#                     self.image = (self.image * 255).astype(np.uint8)
#                 else:
#                     self.image = ((self.image - self.image.min()) / 
#                                 (self.image.max() - self.image.min()) * 255).astype(np.uint8)
        
#         print(f"Final image shape: {self.image.shape}, dtype: {self.image.dtype}")
#         return self.image
    
#     def detect_vegetation_candidates(self, method="enhanced", min_area=200, max_area=5000):
#         """
#         Detect potential tree locations using vegetation indices
        
#         Args:
#             method: "simple", "enhanced", "ndvi", or "adaptive"
#             min_area: Minimum area for vegetation blobs
#             max_area: Maximum area for vegetation blobs
#         """
#         h, w = self.image.shape[:2]
        
#         if method == "ndvi":
#             # Pseudo-NDVI using RGB (Green as NIR substitute)
#             red = self.image[:, :, 0].astype(np.float32)
#             green = self.image[:, :, 1].astype(np.float32)
            
#             ndvi = (green - red) / (green + red + 1e-8)
#             vegetation_mask = ((ndvi > 0.1) * 255).astype(np.uint8)
            
#         elif method == "enhanced":
#             # Enhanced vegetation detection combining multiple indices
#             red = self.image[:, :, 0].astype(np.float32)
#             green = self.image[:, :, 1].astype(np.float32)
#             blue = self.image[:, :, 2].astype(np.float32)
            
#             # Green ratio
#             green_ratio = green / (red + green + blue + 1e-8)
            
#             # Excess green index
#             exg = 2 * green - red - blue
            
#             # Combine indices
#             vegetation_score = (green_ratio > 0.4) & (exg > 10) & (green > red + 5)
#             vegetation_mask = (vegetation_score * 255).astype(np.uint8)
            
#         elif method == "adaptive":
#             # Adaptive thresholding on green channel
#             green = self.image[:, :, 1]
#             vegetation_mask = cv2.adaptiveThreshold(
#                 green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -10
#             )
            
#         else:  # simple
#             green = self.image[:, :, 1]
#             red = self.image[:, :, 0]
#             vegetation_mask = np.where(green > red + 15, 255, 0).astype(np.uint8)
        
#         # Store vegetation mask for visualization
#         self.vegetation_mask = vegetation_mask
        
#         # Morphological operations to clean mask
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
#         vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)
        
#         # Remove small noise
#         kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#         vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel_small)
        
#         # Find contours for tree candidates
#         contours, _ = cv2.findContours(vegetation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         candidates = []
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if min_area < area < max_area:
#                 # Get centroid
#                 M = cv2.moments(contour)
#                 if M["m00"] != 0:
#                     cx = int(M["m10"] / M["m00"])
#                     cy = int(M["m01"] / M["m00"])
                    
#                     # Add some quality metrics
#                     x, y, w_cnt, h_cnt = cv2.boundingRect(contour)
#                     aspect_ratio = w_cnt / h_cnt
#                     extent = area / (w_cnt * h_cnt)
                    
#                     # Filter by shape (trees are usually somewhat round)
#                     if 0.3 < aspect_ratio < 3.0 and extent > 0.3:
#                         candidates.append({
#                             'center': [cx, cy],
#                             'area': area,
#                             'aspect_ratio': aspect_ratio,
#                             'extent': extent
#                         })
        
#         print(f"Found {len(candidates)} vegetation candidates using {method} method")
#         return candidates, vegetation_mask
    
#     def simple_segmentation(self, candidates, expansion_factor=1.2):
#         """
#         Simple segmentation around detected candidates (mock SAM behavior)
#         """
#         segments = []
#         h, w = self.image.shape[:2]
        
#         for candidate in candidates:
#             cx, cy = candidate['center']
#             area = candidate['area']
            
#             # Estimate radius from area
#             radius = int(np.sqrt(area / np.pi) * expansion_factor)
#             radius = max(10, min(radius, 50))  # Reasonable bounds
            
#             # Create circular mask
#             mask = np.zeros((h, w), dtype=bool)
#             y, x = np.ogrid[:h, :w]
#             mask_condition = (x - cx)**2 + (y - cy)**2 <= radius**2
            
#             # Ensure mask is within image bounds
#             mask[mask_condition] = True
            
#             segments.append({
#                 'mask': mask,
#                 'center': [cx, cy],
#                 'area': np.sum(mask),
#                 'score': 0.8 + np.random.rand() * 0.2,  # Mock confidence
#                 'radius': radius
#             })
        
#         self.segments = segments
#         return segments
    
#     def filter_segments(self, min_area=300, max_area=8000, overlap_threshold=0.4):
#         """Filter segments by area and remove overlaps"""
#         if not self.segments:
#             return []
        
#         # Filter by area
#         filtered = [seg for seg in self.segments if min_area <= seg['area'] <= max_area]
        
#         # Remove overlapping segments (keep higher score)
#         final_segments = []
        
#         for i, seg1 in enumerate(filtered):
#             keep = True
#             mask1 = seg1['mask']
            
#             for j, seg2 in enumerate(final_segments):
#                 mask2 = seg2['mask']
                
#                 # Calculate overlap
#                 intersection = np.sum(mask1 & mask2)
#                 union = np.sum(mask1 | mask2)
#                 overlap_ratio = intersection / union if union > 0 else 0
                
#                 if overlap_ratio > overlap_threshold:
#                     # Keep segment with higher score
#                     if seg1['score'] <= seg2['score']:
#                         keep = False
#                         break
#                     else:
#                         # Remove the existing segment
#                         final_segments.remove(seg2)
            
#             if keep:
#                 final_segments.append(seg1)
        
#         self.segments = final_segments
#         return final_segments
    
#     def cluster_tree_centers(self, eps=25, min_samples=1):
#         """Get final tree centers by clustering"""
#         if not self.segments:
#             return []
        
#         centers = np.array([seg['center'] for seg in self.segments])
        
#         if len(centers) <= 1:
#             self.tree_centers = centers.tolist()
#             return self.tree_centers
        
#         # Apply DBSCAN clustering
#         clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
#         labels = clustering.labels_
        
#         # Get representative center for each cluster
#         clustered_centers = []
#         for label in set(labels):
#             if label == -1:  # Noise
#                 continue
            
#             cluster_points = centers[labels == label]
#             center = np.mean(cluster_points, axis=0)
#             clustered_centers.append(center.tolist())
        
#         self.tree_centers = clustered_centers
#         return clustered_centers
    
#     def count_trees(self, vegetation_method="enhanced", min_area=200, max_area=5000):
#         """Complete pipeline to count trees in RGB TIF"""
#         if self.image is None:
#             raise ValueError("No image loaded. Call load_rgb_tif() first.")
        
#         print("🌱 Starting tree detection pipeline...")
        
#         # Step 1: Detect vegetation candidates
#         candidates, veg_mask = self.detect_vegetation_candidates(
#             method=vegetation_method, 
#             min_area=min_area, 
#             max_area=max_area
#         )
        
#         if not candidates:
#             print("No vegetation candidates found!")
#             return 0, []
        
#         # Step 2: Create segments around candidates
#         print("🔍 Creating segments...")
#         segments = self.simple_segmentation(candidates)
        
#         # Step 3: Filter segments
#         print("🔧 Filtering segments...")
#         filtered_segments = self.filter_segments()
        
#         # Step 4: Get final tree centers
#         print("🌳 Clustering tree centers...")
#         tree_centers = self.cluster_tree_centers()
        
#         print(f"✅ Detection complete! Found {len(tree_centers)} trees")
#         return len(tree_centers), tree_centers
    
#     def visualize_results(self, save_path=None):
#         """Create visualization of results"""
#         if self.image is None:
#             return
        
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
#         # Original image
#         axes[0, 0].imshow(self.image)
#         axes[0, 0].set_title('Original RGB TIF')
#         axes[0, 0].axis('off')
        
#         # Vegetation detection
#         if hasattr(self, 'vegetation_mask'):
#             axes[0, 1].imshow(self.vegetation_mask, cmap='Greens')
#             axes[0, 1].set_title('Vegetation Detection')
#             axes[0, 1].axis('off')
#         else:
#             axes[0, 1].imshow(self.image)
#             axes[0, 1].set_title('Processed Image')
#             axes[0, 1].axis('off')
        
#         # Segmentation overlay
#         overlay = self.image.copy()
#         if self.segments:
#             colors = plt.cm.Set3(np.linspace(0, 1, len(self.segments)))
#             for i, seg in enumerate(self.segments):
#                 mask = seg['mask']
#                 color = colors[i][:3]
#                 overlay[mask] = overlay[mask] * 0.7 + np.array(color) * 255 * 0.3
        
#         axes[1, 0].imshow(overlay.astype(np.uint8))
#         axes[1, 0].set_title(f'Detected Segments ({len(self.segments)})')
#         axes[1, 0].axis('off')
        
#         # Tree centers - Fixed scatter plot to avoid matplotlib warning
#         axes[1, 1].imshow(self.image)
#         if self.tree_centers:
#             centers = np.array(self.tree_centers)
#             # Use '+' marker instead of 'x' to avoid edge color issues
#             axes[1, 1].scatter(centers[:, 0], centers[:, 1], 
#                              c='red', s=100, marker='+', linewidths=3)
            
#             # Add numbers with better visibility
#             for i, center in enumerate(centers):
#                 axes[1, 1].annotate(str(i+1), (center[0], center[1]), 
#                                   xytext=(5, 5), textcoords='offset points',
#                                   color='yellow', fontweight='bold', fontsize=10,
#                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
        
#         axes[1, 1].set_title(f'Tree Centers ({len(self.tree_centers)} trees)')
#         axes[1, 1].axis('off')
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             print(f"Visualization saved to {save_path}")
        
#         plt.show()
    
#     def export_results(self, output_path):
#         """Export results to JSON - Fixed to handle CRS serialization"""
#         # Handle metadata serialization properly
#         serializable_metadata = {}
#         for key, value in self.metadata.items():
#             if key == 'crs':
#                 # Convert CRS to string representation
#                 serializable_metadata[key] = str(value) if value is not None else None
#             elif key == 'transform':
#                 # Convert Affine transform to list
#                 if value is not None:
#                     serializable_metadata[key] = list(value)[:6]  # Get the 6 transform parameters
#                 else:
#                     serializable_metadata[key] = None
#             elif key == 'bounds':
#                 # Convert bounds to list
#                 serializable_metadata[key] = list(value) if value is not None else None
#             else:
#                 # For other metadata, try to convert to basic types
#                 try:
#                     json.dumps(value)  # Test if it's serializable
#                     serializable_metadata[key] = value
#                 except (TypeError, ValueError):
#                     serializable_metadata[key] = str(value)
        
#         results = {
#             'tree_count': len(self.tree_centers),
#             'tree_centers': self.tree_centers,
#             'image_info': {
#                 'shape': list(self.image.shape),
#                 'dtype': str(self.image.dtype)
#             },
#             'metadata': serializable_metadata,
#             'segments_info': {
#                 'total_segments': len(self.segments),
#                 'avg_area': np.mean([s['area'] for s in self.segments]) if self.segments else 0,
#                 'avg_score': np.mean([s['score'] for s in self.segments]) if self.segments else 0
#             }
#         }
        
#         with open(output_path, 'w') as f:
#             json.dump(results, f, indent=2)
        
#         print(f"Results exported to {output_path}")

# # Quick function to count trees in your RGB TIF
# def count_trees_in_rgb_tif(tif_path, output_dir="./results"):
#     """
#     Quick function to count trees in an RGB TIF file
    
#     Args:
#         tif_path: Path to your RGB TIF file
#         output_dir: Directory to save results
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Initialize counter
#     counter = RGBTifTreeCounter()
    
#     try:
#         # Load your TIF
#         print(f"📁 Loading {tif_path}...")
#         counter.load_rgb_tif(tif_path, max_size=2048)
        
#         # Count trees
#         tree_count, centers = counter.count_trees(
#             vegetation_method="enhanced",  # Try "ndvi", "adaptive", or "simple" if this doesn't work well
#             min_area=200,  # Adjust based on your image resolution
#             max_area=5000
#         )
        
#         print(f"🌳 Found {tree_count} trees!")
        
#         # Save visualization
#         viz_path = os.path.join(output_dir, "tree_detection_results.png")
#         counter.visualize_results(viz_path)
        
#         # Save results
#         results_path = os.path.join(output_dir, "tree_count_results.json")
#         counter.export_results(results_path)
        
#         return tree_count, centers, counter
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
        
#         # Save results even if something fails
#         if counter and counter.tree_centers:
#             try:
#                 results_path = os.path.join(output_dir, "tree_count_results.json")
#                 counter.export_results(results_path)
#                 print(f"📄 Results saved to {results_path}")
#                 return len(counter.tree_centers), counter.tree_centers, counter
#             except Exception as e2:
#                 print(f"❌ Could not save results: {e2}")
        
#         return 0, [], None

# # Example usage - just run this with your TIF file!
# if __name__ == "__main__":
#     # Replace with your TIF file path
#     tif_file = "Image.tif"
    
#     # Count trees
#     tree_count, tree_centers, counter = count_trees_in_rgb_tif(tif_file)
    
#     print(f"\n🎯 Final Results:")
#     print(f"   Trees detected: {tree_count}")
#     print(f"   Results saved in: ./results/")
    
#     # Try different parameters if results aren't good
#     if tree_count == 0:
#         print("\n💡 Try adjusting parameters:")
#         print("   - Change vegetation_method to 'ndvi', 'adaptive', or 'simple'")
#         print("   - Adjust min_area and max_area based on your tree sizes")
#         print("   - Check if image loaded correctly")





import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
from sklearn.cluster import DBSCAN
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import tifffile
from PIL import Image
from shapely.geometry import Point, Polygon

class RGBTifTreeCounter:
    """Simplified tree counter optimized for RGB TIF files"""
    
    def __init__(self):
        self.image = None
        self.metadata = {}
        self.segments = []
        self.tree_centers = []
        
    def load_rgb_tif(self, tif_path, max_size=2048):
        """
        Load RGB TIF file with automatic resizing if needed
        
        Args:
            tif_path: Path to RGB TIF file
            max_size: Maximum dimension to resize to if image is larger
        """
        print(f"Loading RGB TIF: {tif_path}")
        
        try:
            # Try rasterio first for geospatial TIFs
            with rasterio.open(tif_path) as src:
                self.metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': str(src.dtypes[0])
                }
                
                print(f"TIF info: {src.count} bands, {src.width}x{src.height}, {src.dtypes[0]}")
                
                # Check if we need to resize
                if src.width > max_size or src.height > max_size:
                    scale = max_size / max(src.width, src.height)
                    new_width = int(src.width * scale)
                    new_height = int(src.height * scale)
                    
                    print(f"Resizing from {src.width}x{src.height} to {new_width}x{new_height}")
                    
                    # Read RGB bands with resampling
                    rgb_bands = [1, 2, 3] if src.count >= 3 else [1]
                    self.image = src.read(
                        rgb_bands,
                        out_shape=(len(rgb_bands), new_height, new_width),
                        resampling=Resampling.bilinear
                    )
                    self.image = np.transpose(self.image, (1, 2, 0))
                else:
                    # Read RGB bands at full resolution
                    rgb_bands = [1, 2, 3] if src.count >= 3 else [1]
                    self.image = src.read(rgb_bands)
                    self.image = np.transpose(self.image, (1, 2, 0))
                
                # Handle single band (convert to RGB)
                if len(self.image.shape) == 2 or self.image.shape[2] == 1:
                    if len(self.image.shape) == 2:
                        self.image = np.stack([self.image, self.image, self.image], axis=-1)
                    else:
                        self.image = np.repeat(self.image, 3, axis=2)
        
        except Exception as e:
            print(f"Rasterio failed: {e}. Trying alternative methods...")
            
            # Fallback to tifffile
            try:
                self.image = tifffile.imread(tif_path)
                
                if len(self.image.shape) == 3 and self.image.shape[0] < self.image.shape[2]:
                    self.image = np.transpose(self.image, (1, 2, 0))
                
                # Take first 3 channels
                if len(self.image.shape) == 3 and self.image.shape[2] > 3:
                    self.image = self.image[:, :, :3]
                elif len(self.image.shape) == 2:
                    self.image = np.stack([self.image, self.image, self.image], axis=-1)
                
                # Resize if needed
                if max(self.image.shape[:2]) > max_size:
                    h, w = self.image.shape[:2]
                    scale = max_size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    self.image = cv2.resize(self.image, (new_w, new_h))
                    
            except Exception as e2:
                print(f"Tifffile failed: {e2}. Trying PIL...")
                
                # Final fallback
                with Image.open(tif_path) as img:
                    # Resize if needed
                    if max(img.size) > max_size:
                        scale = max_size / max(img.size)
                        new_size = (int(img.width * scale), int(img.height * scale))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    self.image = np.array(img)
                    
                    if len(self.image.shape) == 2:
                        self.image = np.stack([self.image, self.image, self.image], axis=-1)
        
        # Normalize to uint8
        if self.image.dtype != np.uint8:
            if self.image.dtype == np.uint16:
                self.image = (self.image / 256).astype(np.uint8)
            elif self.image.dtype in [np.float32, np.float64]:
                if self.image.max() <= 1.0:
                    self.image = (self.image * 255).astype(np.uint8)
                else:
                    self.image = ((self.image - self.image.min()) / 
                                (self.image.max() - self.image.min()) * 255).astype(np.uint8)
        
        print(f"Final image shape: {self.image.shape}, dtype: {self.image.dtype}")
        return self.image
    
    def detect_vegetation_candidates(self, method="enhanced", min_area=200, max_area=5000):
        """
        Detect potential tree locations using vegetation indices
        
        Args:
            method: "simple", "enhanced", "ndvi", or "adaptive"
            min_area: Minimum area for vegetation blobs
            max_area: Maximum area for vegetation blobs
        """
        h, w = self.image.shape[:2]
        
        if method == "ndvi":
            # Pseudo-NDVI using RGB (Green as NIR substitute)
            red = self.image[:, :, 0].astype(np.float32)
            green = self.image[:, :, 1].astype(np.float32)
            
            ndvi = (green - red) / (green + red + 1e-8)
            vegetation_mask = ((ndvi > 0.1) * 255).astype(np.uint8)
            
        elif method == "enhanced":
            # Enhanced vegetation detection combining multiple indices
            red = self.image[:, :, 0].astype(np.float32)
            green = self.image[:, :, 1].astype(np.float32)
            blue = self.image[:, :, 2].astype(np.float32)
            
            # Green ratio
            green_ratio = green / (red + green + blue + 1e-8)
            
            # Excess green index
            exg = 2 * green - red - blue
            
            # Combine indices
            vegetation_score = (green_ratio > 0.4) & (exg > 10) & (green > red + 5)
            vegetation_mask = (vegetation_score * 255).astype(np.uint8)
            
        elif method == "adaptive":
            # Adaptive thresholding on green channel
            green = self.image[:, :, 1]
            vegetation_mask = cv2.adaptiveThreshold(
                green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -10
            )
            
        else:  # simple
            green = self.image[:, :, 1]
            red = self.image[:, :, 0]
            vegetation_mask = np.where(green > red + 15, 255, 0).astype(np.uint8)
        
        # Store vegetation mask for visualization
        self.vegetation_mask = vegetation_mask
        
        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Find contours for tree candidates
        contours, _ = cv2.findContours(vegetation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Add some quality metrics
                    x, y, w_cnt, h_cnt = cv2.boundingRect(contour)
                    aspect_ratio = w_cnt / h_cnt
                    extent = area / (w_cnt * h_cnt)
                    
                    # Filter by shape (trees are usually somewhat round)
                    if 0.3 < aspect_ratio < 3.0 and extent > 0.3:
                        candidates.append({
                            'center': [cx, cy],
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'extent': extent
                        })
        
        print(f"Found {len(candidates)} vegetation candidates using {method} method")
        return candidates, vegetation_mask
    
    def simple_segmentation(self, candidates, expansion_factor=1.2):
        """
        Simple segmentation around detected candidates (mock SAM behavior)
        """
        segments = []
        h, w = self.image.shape[:2]
        
        for candidate in candidates:
            cx, cy = candidate['center']
            area = candidate['area']
            
            # Estimate radius from area
            radius = int(np.sqrt(area / np.pi) * expansion_factor)
            radius = max(10, min(radius, 50))  # Reasonable bounds
            
            # Create circular mask
            mask = np.zeros((h, w), dtype=bool)
            y, x = np.ogrid[:h, :w]
            mask_condition = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            # Ensure mask is within image bounds
            mask[mask_condition] = True
            
            segments.append({
                'mask': mask,
                'center': [cx, cy],
                'area': np.sum(mask),
                'score': 0.8 + np.random.rand() * 0.2,  # Mock confidence
                'radius': radius
            })
        
        self.segments = segments
        return segments
    
    def filter_segments(self, min_area=300, max_area=8000, overlap_threshold=0.4):
        """Filter segments by area and remove overlaps"""
        if not self.segments:
            return []
        
        # Filter by area
        filtered = [seg for seg in self.segments if min_area <= seg['area'] <= max_area]
        
        # Remove overlapping segments (keep higher score)
        final_segments = []
        
        for i, seg1 in enumerate(filtered):
            keep = True
            mask1 = seg1['mask']
            
            for j, seg2 in enumerate(final_segments):
                mask2 = seg2['mask']
                
                # Calculate overlap
                intersection = np.sum(mask1 & mask2)
                union = np.sum(mask1 | mask2)
                overlap_ratio = intersection / union if union > 0 else 0
                
                if overlap_ratio > overlap_threshold:
                    # Keep segment with higher score
                    if seg1['score'] <= seg2['score']:
                        keep = False
                        break
                    else:
                        # Remove the existing segment
                        final_segments.remove(seg2)
            
            if keep:
                final_segments.append(seg1)
        
        self.segments = final_segments
        return final_segments
    
    def cluster_tree_centers(self, eps=25, min_samples=1):
        """Get final tree centers by clustering"""
        if not self.segments:
            return []
        
        centers = np.array([seg['center'] for seg in self.segments])
        
        if len(centers) <= 1:
            self.tree_centers = centers.tolist()
            return self.tree_centers
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
        labels = clustering.labels_
        
        # Get representative center for each cluster
        clustered_centers = []
        for label in set(labels):
            if label == -1:  # Noise
                continue
            
            cluster_points = centers[labels == label]
            center = np.mean(cluster_points, axis=0)
            clustered_centers.append(center.tolist())
        
        self.tree_centers = clustered_centers
        return clustered_centers
    
    def count_trees(self, vegetation_method="enhanced", min_area=200, max_area=5000):
        """Complete pipeline to count trees in RGB TIF"""
        if self.image is None:
            raise ValueError("No image loaded. Call load_rgb_tif() first.")
        
        print("🌱 Starting tree detection pipeline...")
        
        # Step 1: Detect vegetation candidates
        candidates, veg_mask = self.detect_vegetation_candidates(
            method=vegetation_method, 
            min_area=min_area, 
            max_area=max_area
        )
        
        if not candidates:
            print("No vegetation candidates found!")
            return 0, []
        
        # Step 2: Create segments around candidates
        print("🔍 Creating segments...")
        segments = self.simple_segmentation(candidates)
        
        # Step 3: Filter segments
        print("🔧 Filtering segments...")
        filtered_segments = self.filter_segments()
        
        # Step 4: Get final tree centers
        print("🌳 Clustering tree centers...")
        tree_centers = self.cluster_tree_centers()
        
        print(f"✅ Detection complete! Found {len(tree_centers)} trees")
        return len(tree_centers), tree_centers
    
    def visualize_results(self, save_path=None):
        """Create visualization of results"""
        if self.image is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(self.image)
        axes[0, 0].set_title('Original RGB TIF')
        axes[0, 0].axis('off')
        
        # Vegetation detection
        if hasattr(self, 'vegetation_mask'):
            axes[0, 1].imshow(self.vegetation_mask, cmap='Greens')
            axes[0, 1].set_title('Vegetation Detection')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].imshow(self.image)
            axes[0, 1].set_title('Processed Image')
            axes[0, 1].axis('off')
        
        # Segmentation overlay
        overlay = self.image.copy()
        if self.segments:
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.segments)))
            for i, seg in enumerate(self.segments):
                mask = seg['mask']
                color = colors[i][:3]
                overlay[mask] = overlay[mask] * 0.7 + np.array(color) * 255 * 0.3
        
        axes[1, 0].imshow(overlay.astype(np.uint8))
        axes[1, 0].set_title(f'Detected Segments ({len(self.segments)})')
        axes[1, 0].axis('off')
        
        # Tree centers - Fixed scatter plot to avoid matplotlib warning
        axes[1, 1].imshow(self.image)
        if self.tree_centers:
            centers = np.array(self.tree_centers)
            # Use '+' marker instead of 'x' to avoid edge color issues
            axes[1, 1].scatter(centers[:, 0], centers[:, 1], 
                             c='red', s=100, marker='+', linewidths=3)
            
            # Add numbers with better visibility
            for i, center in enumerate(centers):
                axes[1, 1].annotate(str(i+1), (center[0], center[1]), 
                                  xytext=(5, 5), textcoords='offset points',
                                  color='yellow', fontweight='bold', fontsize=10,
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
        
        axes[1, 1].set_title(f'Tree Centers ({len(self.tree_centers)} trees)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def export_results(self, output_path):
        """Export results to JSON - Fixed to handle CRS serialization"""
        # Handle metadata serialization properly
        serializable_metadata = {}
        for key, value in self.metadata.items():
            if key == 'crs':
                # Convert CRS to string representation
                serializable_metadata[key] = str(value) if value is not None else None
            elif key == 'transform':
                # Convert Affine transform to list
                if value is not None:
                    serializable_metadata[key] = list(value)[:6]  # Get the 6 transform parameters
                else:
                    serializable_metadata[key] = None
            elif key == 'bounds':
                # Convert bounds to list
                serializable_metadata[key] = list(value) if value is not None else None
            else:
                # For other metadata, try to convert to basic types
                try:
                    json.dumps(value)  # Test if it's serializable
                    serializable_metadata[key] = value
                except (TypeError, ValueError):
                    serializable_metadata[key] = str(value)
        
        results = {
            'tree_count': len(self.tree_centers),
            'tree_centers': self.tree_centers,
            'image_info': {
                'shape': list(self.image.shape),
                'dtype': str(self.image.dtype)
            },
            'metadata': serializable_metadata,
            'segments_info': {
                'total_segments': len(self.segments),
                'avg_area': np.mean([s['area'] for s in self.segments]) if self.segments else 0,
                'avg_score': np.mean([s['score'] for s in self.segments]) if self.segments else 0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {output_path}")

# Quick function to count trees in your RGB TIF
def count_trees_in_rgb_tif(tif_path, output_dir="./results"):
    """
    Quick function to count trees in an RGB TIF file
    
    Args:
        tif_path: Path to your RGB TIF file
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize counter
    counter = RGBTifTreeCounter()
    
    try:
        # Load your TIF
        print(f"📁 Loading {tif_path}...")
        counter.load_rgb_tif(tif_path, max_size=2048)
        
        # Count trees
        tree_count, centers = counter.count_trees(
            vegetation_method="ndvi",  # Try "ndvi", "adaptive", or "simple" if this doesn't work well "enhanced"
            min_area=200,  # Adjust based on your image resolution
            max_area=5000
        )
        
        print(f"🌳 Found {tree_count} trees!")
        
        # Save visualization
        viz_path = os.path.join(output_dir, "tree_detection_results.png")
        counter.visualize_results(viz_path)
        
        # Save results
        results_path = os.path.join(output_dir, "tree_count_results.json")
        counter.export_results(results_path)
        
        return tree_count, centers, counter
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Save results even if something fails
        if counter and counter.tree_centers:
            try:
                results_path = os.path.join(output_dir, "tree_count_results.json")
                counter.export_results(results_path)
                print(f"📄 Results saved to {results_path}")
                return len(counter.tree_centers), counter.tree_centers, counter
            except Exception as e2:
                print(f"❌ Could not save results: {e2}")
        
        return 0, [], None

# Example usage - just run this with your TIF file!
if __name__ == "__main__":
    # Replace with your TIF file path
    tif_file = "Image.tif"
    
    # Count trees
    tree_count, tree_centers, counter = count_trees_in_rgb_tif(tif_file)
    
    print(f"\n🎯 Final Results:")
    print(f"   Trees detected: {tree_count}")
    print(f"   Results saved in: ./results/")
    
    # Try different parameters if results aren't good
    if tree_count == 0:
        print("\n💡 Try adjusting parameters:")
        print("   - Change vegetation_method to 'ndvi', 'adaptive', or 'simple'")
        print("   - Adjust min_area and max_area based on your tree sizes")
        print("   - Check if image loaded correctly")