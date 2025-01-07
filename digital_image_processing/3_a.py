#3(a): Take grayscale image of size 512x512, apply average & median spatial filters with 5x5 mask & observe their performance for noise suppression in term of PSNR

import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def psnr(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:  # Perfect match
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Load and resize the grayscale image to 512x512
image_path = 'cat.jpeg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))

# Add Gaussian noise
noisy_image = add_gaussian_noise(image)

# Apply average filtering (5x5 mask)
average_filtered = cv2.blur(noisy_image, (5, 5))

# Apply median filtering (5x5 mask)
median_filtered = cv2.medianBlur(noisy_image, 5)

# Calculate PSNR for both filters
psnr_avg = psnr(image, average_filtered)
psnr_median = psnr(image, median_filtered)

# Plot the results
plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Noisy Image
plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title("Noisy Image")
plt.axis('off')

# Average Filtered Image
plt.subplot(2, 2, 3)
plt.imshow(average_filtered, cmap='gray')
plt.title(f"Average Filtered (PSNR: {psnr_avg:.2f} dB)")
plt.axis('off')

# Median Filtered Image
plt.subplot(2, 2, 4)
plt.imshow(median_filtered, cmap='gray')
plt.title(f"Median Filtered (PSNR: {psnr_median:.2f} dB)")
plt.axis('off')

plt.tight_layout()
plt.show()
