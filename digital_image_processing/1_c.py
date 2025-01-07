
#1(c)Take grayscale image of size 512x512 and perform the following operations:
#Illustrate the histogram of the image and make single threshold segmentation observed from the histogram


import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_histogram(image):
    gray_levels_count = np.zeros(256)
    height, width = image.shape

    for r in range (height):
        for c in range(width):
            gray_levels_count[image[r, c]] += 1

    plt.bar(range(256), gray_levels_count, width = 1.0, color = "gray")
    plt.title("The Histogram of the Image")
    plt.show()

original_image = cv2.imread('rose.jpg.tif', cv2.IMREAD_GRAYSCALE)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("The Skull Image")
plt.show()


threshold_intensity = 27
segmented_image = np.where(original_image < threshold_intensity, 0, 255)
segmented_image = np.uint8(segmented_image)

plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title("The Segmented Image")
plt.show()

generate_histogram(segmented_image)