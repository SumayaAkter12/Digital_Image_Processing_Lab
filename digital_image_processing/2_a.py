
#2(a): Take a grayscale image of size 512x512, perform the brightness enhancement of a specific range of gray levels & observe its result

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and resize the grayscale image to 512x512
image_path = 'rose.jpg.tif'  # Replace with the image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))

# Define the range of gray levels to enhance
lower_bound, upper_bound = 100, 150
enhancement_value = 50  # Brightness increase

# Create a copy of the image for modification
enhanced_image = image.copy()

# Apply brightness enhancement to the specified range
mask = (image >= lower_bound) & (image <= upper_bound)
enhanced_image[mask] = np.clip(image[mask] + enhancement_value, 0, 255)

# Plot the original and enhanced images
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title("Original Image")
plt.axis('off')

# Enhanced Image
plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap='gray', vmin=0, vmax=255)
plt.title(f"Enhanced Image\nRange: {lower_bound}-{upper_bound}")
plt.axis('off')

plt.tight_layout()
plt.show()
