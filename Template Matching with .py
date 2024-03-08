from skimage import feature

def detect_objects_hog(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    return hogImage

# Example usage:
image_path = "image.jpg"
hog_image = detect_objects_hog(image_path)
cv2.imshow("HOG Image", hog_image)
cv2.waitKey(0)
