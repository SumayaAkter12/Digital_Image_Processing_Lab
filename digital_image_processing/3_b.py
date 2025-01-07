#3(b): Take grayscale image of size 512x512, use different size of mask (3x3, 5x5, 7x7) with average filter for noise suppression & observe their performance in term of PSNR

import cv2
import numpy as np 
import matplotlib.pyplot as plt 

originalImage = cv2.imread('Characters_Test.tif', cv2.IMREAD_GRAYSCALE)
#originalImage = cv2.resize(originalImage, (512, 512))

def addSaltPeperNoise(image, saltRatio, pepperRatio):
    noisyImage = image.copy()
    numOfPixels = image.size
    numOfSaltPixels = int(saltRatio * numOfPixels)
    numOfPepperPixels = int(pepperRatio * numOfPixels)

    for i in range(numOfSaltPixels):
        x, y = np.random.randint(0, noisyImage.shape) # (row, col) = (512, 512)
        noisyImage[x][y] = 255
    
    for i in range(numOfPepperPixels):
        x, y = np.random.randint(0, noisyImage.shape) # (row, col) = (512, 512)
        noisyImage[x][y] = 0

    return noisyImage

def averageFilter(image, kernelSize):
    padSize = kernelSize // 2
    paddedImage = np.pad(image, (padSize, padSize), mode='constant')

    filteredImage = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernelRegion = paddedImage[i: i+kernelSize, j: j+kernelSize]
            average = np.mean(kernelRegion)
            filteredImage[i, j] = average
    
    return filteredImage

# Peak Signal-to-Noise Ratio
def calcPNSR(image1, image2):  
    image1, image2 = np.float64(image1), np.float64(image2)
    mse = np.mean((image1 - image2) ** 2)   # Mean Squared Error (MSE)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)

    print(f"comute pnsr: {20 * np.log10(255.0 / np.sqrt(mse))}")
    
    return round(psnr, 2)

plt.figure(figsize=(10, 12))

plt.subplot(3, 2, 1)
plt.imshow(originalImage, cmap='gray')
plt.title(f"Original Image (PNSR={calcPNSR(originalImage, originalImage)})")

noisyImage = addSaltPeperNoise(originalImage, 0.05, 0.15)
plt.subplot(3, 2, 2)
plt.imshow(noisyImage, cmap='gray')
plt.title(f"Noisy Image (Salt & Pepper) (PNSR={calcPNSR(originalImage, noisyImage)})")

averageFilteredImage = averageFilter(noisyImage, 3)
plt.subplot(3, 2, 3)
plt.imshow(averageFilteredImage, cmap='gray')
plt.title(f"Filtered Image (PNSR={calcPNSR(originalImage, averageFilteredImage)}) - mask (3x3)")

averageFilteredImage = averageFilter(noisyImage, 5)
plt.subplot(3, 2, 4)
plt.imshow(averageFilteredImage, cmap='gray')
plt.title(f"Filtered Image (PNSR={calcPNSR(originalImage, averageFilteredImage)}) - mask (5x5)")

averageFilteredImage = averageFilter(noisyImage, 7)
plt.subplot(3, 2, 5)
plt.imshow(averageFilteredImage, cmap='gray')
plt.title(f"Filtered Image (PNSR={calcPNSR(originalImage, averageFilteredImage)}) - mask (7x7)")
plt.show()