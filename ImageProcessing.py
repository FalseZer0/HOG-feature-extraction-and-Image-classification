import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math
from cv2 import cv2

def mat2gray(A):
    A = np.double(A)
    out = np.zeros(A.shape, np.double)
    normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
    return normalized
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
A = plt.imread('test_color.jpg')
A_gray = rgb2gray(A)
nc, nr = A_gray.shape
Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # Sobel operator
Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) # Sobel operator
Ax = signal.correlate2d(A_gray, Sx, mode='same')
Ay = signal.correlate2d(A_gray, Sy, mode='same')
plt.figure(num="Image")
a = plt.subplot(2,2,1); plt.imshow(A); plt.axis('off'); a.title.set_text("Original")
b = plt.subplot(2,2,2); plt.imshow(A_gray, cmap='gray'); plt.axis('off'); b.title.set_text("Gray")
c = plt.subplot(2,2,3); plt.imshow(np.uint8(mat2gray(Ax)*255), cmap='gray');plt.axis('off')
c.title.set_text("Result of horizontal gradient")
d = plt.subplot(2,2,4); plt.imshow(np.uint8(mat2gray(Ay)*255), cmap='gray')
d.title.set_text("Result of vertical gradient")
plt.axis('off')
plt.show()
print(str(A.shape )+ " size of original image\n"+str(A_gray.shape)+" size of the gray image")


