#1(a)Take grayscale image of size 512x512 and perform the following operations:
#Decrease its spatial resolution by half every time and observe its change when displaying in the same window size.



import cv2
import numpy as np
import matplotlib.pyplot as plt

def decrease_resolution(image):
    height, width = image.shape
    decreased_image = np.zeros((height // 2, width // 2))

    for r in range(0, height, 2):
        for c in range(0, width, 2):
            decreased_image[r // 2, c // 2] = image[r, c]
        
    return np.uint8(decreased_image)




original_image = cv2.imread('a.jpg', cv2.IMREAD_GRAYSCALE)
original_image = cv2.resize(original_image, (512, 512))
decreased_image = original_image.copy()
plt.figure(figsize = (13, 13)) 






for k in range (1, 5):
    plt.subplot(2, 2, k)
    plt.imshow(decreased_image, cmap = 'gray')
    height, width = decreased_image.shape
    plt.title(f"{height}x{width}")
    decreased_image = decrease_resolution(decreased_image)

plt.show()