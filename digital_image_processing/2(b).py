import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and resize the grayscale image to 512x512
image_path = 'a.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))

# Normalize the image to range [0, 1] for transformations
image_normalized = image / 255.0

# Power Law Transformation
gamma = 0.5  # Change this to adjust brightness
power_law_transformed = np.clip(1 * (image_normalized ** gamma), 0, 1)

# Inverse Logarithmic Transformation
c = 1  # Scaling constant
inverse_log_transformed = np.clip(c * (np.exp(image_normalized) - 1), 0, 1)

# Scale back to 8-bit range (0-255)
power_law_result = (power_law_transformed * 255).astype(np.uint8)
inverse_log_result = (inverse_log_transformed * 255).astype(np.uint8)

# Plot Original and Transformed Images
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title("Original Image")
plt.axis('off')

# Power Law Transformation
plt.subplot(1, 3, 2)
plt.imshow(power_law_result, cmap='gray', vmin=0, vmax=255)
plt.title("Power Law Transformation")
plt.axis('off')

# Inverse Logarithmic Transformation
plt.subplot(1, 3, 3)
plt.imshow(inverse_log_result, cmap='gray', vmin=0, vmax=255)
plt.title("Inverse Logarithmic Transformation")
plt.axis('off')

plt.tight_layout()
plt.show()
