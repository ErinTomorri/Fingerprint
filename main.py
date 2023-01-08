import cv2
import numpy as np
from skimage.morphology import skeletonize, thin

def remove_dot(invert_thin):
    temp = np.array(invert_thin, dtype=np.uint8)
    temp = temp / 255
    enhanced_img = np.array(temp)
    filter_ = np.zeros((10, 10))
    h, w = temp.shape[:2]
    filter_size = 6

    for i in range(h - filter_size):
        for j in range(w - filter_size):
            filter_ = temp[i:i + filter_size, j:j + filter_size]
            if sum(filter_[:, 0]) == 0 and sum(filter_[:, filter_size - 1]) == 0 and sum(filter_[0, :]) == 0 and sum(filter_[filtersize - 1, :]) == 0:
                temp[i:i + filter_size, j:j + filter_size] = np.zeros((filter_size, filter_size))

    return temp

def get_descriptors(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = np.array(img, dtype=np.uint8)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img[img == 255] = 1
    skeleton = skeletonize(img)
    skeleton = np.array(skeleton, dtype=np.uint8)
    skeleton = remove_dot(skeleton)
    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125
    keypoints = []
    for x in range(harris_normalized.shape[0]):
        for y in range(harris_normalized.shape[1]):
            if harris_normalized[x][y] > threshold_harris:
                keypoints.append(cv2.KeyPoint(y, x, 1))
    orb = cv2.ORB_create()
    _, des = orb.compute(img, keypoints)
    return keypoints, des

def main():
	image_name = sys.argv[1]
	img1 = cv2.imread("database/" + image_name, cv2.IMREAD_GRAYSCALE)
	kp1, des1 = get_descriptors(img1)

	image_name = sys.argv[2]
	img2 = cv2.imread("database/" + image_name, cv2.IMREAD_GRAYSCALE)
	kp2, des2 = get_descriptors(img2)

	# Matching between descriptors
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = sorted(bf.match(des1, des2), key= lambda match:match.distance)
	# Plot keypoints
	img4 = cv2.drawKeypoints(img1, kp1, outImage=None)
	img5 = cv2.drawKeypoints(img2, kp2, outImage=None)
	f, axarr = plt.subplots(1,2)
	axarr[0].imshow(img4)
	axarr[1].imshow(img5)
	plt.show()
	# Plot matches
	img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
	plt.imshow(img3)
	plt.show()

	# Calculate score
	score = 0
	for match in matches:
		score += match.distance
	score_threshold = 33
	if score/len(matches) < score_threshold:
		print("Fingerprint matches.")
	else:
		print("Fingerprint does not match.")



if __name__ == "__main__":
	try:
		main()
	except:
		raise
