# --  kmeans packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
# import utils
import cv2

def dom_colors(import_img, num_colors=1, use_BGR=False, num_clusters=3):
	"""find dominant colors of image via k-means clustering"""
	# reshape image to be a list of pixels
	image = import_img.reshape((import_img.shape[0] * import_img.shape[1], 3))
	# cluster the pixel intensities 
	clt = KMeans(n_clusters = num_clusters)
	clt.fit(image)
	# return as many of dominant colors as required, either as list of tuples or as single tuple
	dom = [tuple(bgr) for bgr in clt.cluster_centers_[:num_colors]] if num_colors > 1 else tuple(clt.cluster_centers_[0])
	if not use_BGR:
		# switch BGR tuple order to RGB in list comprehension, to play nice with skimage functions
		dom = [bgr[::-1] for bgr in dom] if num_colors > 1 else dom[::-1]
	return dom

def remove_color(colored_pic, rgb_color, tolerance=0.7):
	"""blanks out any color in image within tolerance-percentage of color given"""
	# surprisingly, a high tolerance works best for the training pic...
	img = colored_pic.copy()
	# create color tolerance limits based on rgb color
	rlims,glims,blims = ((rgb_color[i]*(1.0-tolerance),rgb_color[i]*(1+tolerance)) for i in range(3))
	# set to black where within tolerance limits
	# rgb stored as [[(255,255,255), (0,0,0)], [(100,100,100), (5,5,5)], etc...]
	img[((img[:, :, 0]>rlims[0]) & (img[:, :, 0]<rlims[1])) & 
	((img[:, :, 1]>glims[0]) & (img[:, :, 1]<glims[1])) &
	((img[:, :, 2]>blims[0]) & (img[:, :, 2]<blims[1]))] = 255
	return img

if __name__ == "__main__":
	# print dom_colors("sc_cropped.png",2,3)
	pass

#  starting point article:
# http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/