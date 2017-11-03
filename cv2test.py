import cv2
import matplotlib.pyplot as plt

img = cv2.imread('ant_img/tandem/0.png', 0)
ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.imshow(th, 'gray')
plt.show()
