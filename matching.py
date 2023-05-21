from skimage import feature, io, color

# Load the image
image = io.imread('path_to_your_image.png')

# Convert the image to grayscale
image = color.rgb2gray(image)

# Compute the HOG features
hog_features = feature.hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

print(hog_features)

from scipy.spatial import distance

# Compute the Euclidean distance between the HOG features of two images
dist = distance.euclidean(hog_features1, hog_features2)

print(dist)
