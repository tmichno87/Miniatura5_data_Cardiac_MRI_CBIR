import cv2
import sys
import numpy as np
from sklearn.cluster import DBSCAN

# Load the image
image_path = sys.argv[1]
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# K-means clustering
Z = np.float32(gray)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
kmeans_image = res.reshape((gray.shape))

# Normalize the image data to 0-1 range
normalized_image = gray / 255.0
pixels = normalized_image.reshape(-1, 1)

# DBSCAN clustering (commented out for now)
# db = DBSCAN(eps=0.05, min_samples=5).fit(pixels)
# labels = db.labels_.reshape(gray.shape)
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(f'Estimated number of clusters: {n_clusters}')
# dbscan_image = np.zeros_like(gray, dtype=np.uint8)
# unique_labels = set(labels)
# for label in unique_labels:
#     if label == -1:
#         gray_level = 0  # Black for noise
#     else:
#         gray_level = int(255 * (label + 1) / n_clusters)
#     dbscan_image[labels == label] = gray_level

# OTSU thresholding
_, otsu_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Mean Shift filtering
meanshift_image = cv2.pyrMeanShiftFiltering(image, 20, 40)

# Watershed algorithm
_, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
markers = cv2.connectedComponents(binary_image)[1]
markers = cv2.watershed(image, markers)

# Convert the markers image to a displayable format
# Markers will be in the range [-1, num_segments], so we normalize it
markers = markers.astype(np.int32)
markers_display = np.zeros_like(markers, dtype=np.uint8)
markers_display[markers == -1] = 255  # Boundary marked with -1
unique_markers = np.unique(markers)
for marker in unique_markers:
    if marker != -1:
        markers_display[markers == marker] = int(255 * marker / len(unique_markers))

# Show each image in separate windows
cv2.imshow('Original', gray)
cv2.imshow('K-Means', kmeans_image)
# cv2.imshow('DBSCAN', dbscan_image)
cv2.imshow('OTSU', otsu_image)
cv2.imshow('Mean Shift', meanshift_image)
cv2.imshow('Watershed', markers_display)
cv2.waitKey(0)
cv2.destroyAllWindows()