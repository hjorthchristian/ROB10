import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# Path to your BlenderProc HDF5 file
file_path = "BlenderProc/output/0.hdf5"  # Change this path if needed

# Open the HDF5 file
with h5py.File(file_path, 'r') as f:
    # List available datasets
    print("Datasets in HDF5 file:")
    f.visit(print)

    # Load RGB image (first frame)
    rgb = np.array(f['colors'])
    print("RGB Image Shape:", rgb.shape)
    rgb_image = np.array(f['colors'][0])  # Shape: (H, W, 3)
    print("RGB Image Shape:", rgb_image.shape)
    print("np.unique(rgb_image):", np.unique(rgb_image))

    # Load Depth map (first frame)
    depth_map = np.array(f['depth'])   # Shape may vary
    print("Depth Map Shape:", depth_map.shape)

if depth_map.ndim == 1:
    # Assume resolution matches RGB image
    height, width  = rgb_image.shape

depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))


# RGB Image
cv.imshow('RGB Image', rgb)
cv.imwrite('output/rgb_image.png', rgb)
cv.waitKey(0)
# Depth Map
# plt.subplot(1, 2, 2)
plt.title('Depth Map')
plt.plot(depth_normalized)
plt.colorbar(label='Normalized Depth')
plt.axis('off')
