
#1(b)Take grayscale image of size 512x512 and perform the following operations:
#Decrease it intensity level resolution by one bit up to reach its binary format observe its change when displaying in the same window size.




import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and resize the grayscale image to 512x512
image = cv2.imread('a.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))

# Plot images with reduced bit depths
plt.figure(figsize=(12, 6))
for bit_depth in range(8, 0, -1):
    reduced_image = (image >> (8 - bit_depth)) << (8 - bit_depth)
    plt.subplot(2, 4, 9 - bit_depth)
    plt.imshow(reduced_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f"{bit_depth} Bits")
    plt.axis('off')

plt.tight_layout()
plt.show()
