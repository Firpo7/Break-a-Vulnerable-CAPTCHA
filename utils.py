import cv2, os, string
import numpy as np

def list_filenames(dir='.', startstr="", endstr=""):
	directory = os.fsencode(dir)
	filenames = []
	for filename in os.listdir(directory):
		filename = filename.decode('ascii')
		if filename.endswith(endstr) and filename.startswith(startstr):
			filenames.append(filename)
	return filenames

def load_data(useTest = False):
	ab = string.ascii_uppercase
	images = []
	values = []
	n=0
	tot = 0
	choosen_set = "test_set" if useTest else "train_set"
	
	for c in ab:
		try:
			filenames = list_filenames('./{}/{}/'.format(choosen_set,c), '', '.bmp')
			tot+=len(filenames)
			for f in filenames:
				im = cv2.imread('./{}/{}/'.format(choosen_set,c) + f, cv2.IMREAD_GRAYSCALE)
				images.append(im)
				values.append(n)
			n+=1
		except FileNotFoundError:
			print("no values for: {}".format(c))

	return np.array(images), values, tot