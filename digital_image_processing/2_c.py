#2(c): Take a grayscale image of size 512x512, find the difference image between the original & the image obtained by last three MSBs
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and resize the grayscale image to 512x512
image_path = 'a.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))

# Extract the last three MSBs
msb_image = (image & 0b11100000)  # Keep only the top 3 bits (MSBs)

# Calculate the difference image
difference_image = cv2.absdiff(image, msb_image)

# Plot the results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title("Original Image")
plt.axis('off')

# MSB-Based Image
plt.subplot(1, 3, 2)
plt.imshow(msb_image, cmap='gray', vmin=0, vmax=255)
plt.title("MSB-Based Image (Top 3 Bits)")
plt.axis('off')

# Difference Image
plt.subplot(1, 3, 3)
plt.imshow(difference_image, cmap='gray', vmin=0, vmax=255)
plt.title("Difference Image")
plt.axis('off')

plt.tight_layout()
plt.show()
