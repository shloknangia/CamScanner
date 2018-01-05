import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./sample-1.jpg')
print image.shape
plt.figure(0)
plt.imshow(image, cmap = 'gray')
plt.figure(12)
# plt.imshow(rotate_bound(image, 180))
# plt.imshow(image)
# plt.show()

image = cv2.resize(image, (1500, 880))
orig = image.copy()
print image.shape
# plt.imshow(image)
# plt.show()


def rectify(h):
	# print h
	h = h.reshape((4, 2))
	# print h
	hnew =  np.zeros((4, 2), dtype = np.float32)
	add = h.sum(1)
	# print "add:", add
	hnew[0] = h[np.argmin(add)]
	hnew[2] = h[np.argmax(add)]

	diff = np.diff(h, axis = 1)
	# print "diff:", diff
	hnew[1] = h[np.argmin(diff)]
	hnew[3] = h[np.argmax(diff)]

	return hnew




# edge detection

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(1, figsize = (7, 7))
plt.imshow(gray, cmap = 'gray')

# blurring , gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# blurred = cv2.medianBlur(gray, 5)
plt.figure(2, figsize = (7, 7))
plt.imshow(blurred, cmap = 'gray')

# applying edge detection
edged = cv2.Canny(blurred, 0, 50)
plt.figure(3, figsize = (7, 7))
plt.imshow(edged, cmap = 'gray')

# plt.show()

# finding largest contour in edgeded image

# find contour
(_, contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# sort the contours by ares in decending order
contours = sorted(contours, key = cv2.contourArea, reverse = True)

# plot boundary against largest one
x, y, w, h = cv2.boundingRect(contours[1])
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.figure(4, figsize = (7, 7))
plt.imshow(image, cmap = 'gray')
# plt.show()

# largest contour with 4 vertices
for c in contours:
	p = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * p, True)

	if len(approx) == 4:
		target = approx
		break

print "largest contour:", target

# plotting it
cv2.drawContours(image, [target], -1, (255, 0, 0), 2)
plt.figure(5, figsize = (7, 7))
plt.imshow(image, cmap = 'gray')
# plt.show()

# mapping target points to a quadrilateral

print "rectify called:"
approx = rectify(target)
print approx

pts2 = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
M = cv2.getPerspectiveTransform(approx, pts2)
dst = cv2.warpPerspective(orig, M, (800, 800))

dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
plt.figure(6, figsize = (7, 7))
plt.imshow(dst, cmap = 'gray')
# plt.show()

# using thresholding on wrapped image to get scanned effect

ret, th1 = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
plt.figure(7, figsize = (7, 7))
plt.imshow(th1, cmap = 'gray')

th2 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
plt.figure(8, figsize = (7, 7))
plt.imshow(th2, cmap = 'gray')

th3 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
plt.figure(9, figsize = (7, 7))
plt.imshow(th3, cmap = 'gray')

ret2,th4 = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure(10, figsize = (7, 7))
plt.imshow(th4, cmap = 'gray')

plt.show()

