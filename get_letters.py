#!/usr/bin/env python3

import requests, sys, os, string, cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter
from utils import *
from termcolor import colored
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w = (255,255,255)
n = (0,0,0)

w1 = 255
n1 = 0

train_set_folder = 'train_set'

def terminate_program(score, N, modelChoosen):
	if modelChoosen:
		print("\n\nAI Score -> {}/{}".format(score, N))
	print(colored("\nSTOP", "red"))
	exit(0)

def save_letters(tmp, s):
	try:
		s = s.upper()
		for n in range(5):
			tmp[n].save('./{}/{}/{}_{}_{}.bmp'.format(train_set_folder,s[n],s[n],n, s))
		print(colored("\tsaved", "green"))
	except Exception as e:
		print(e)
		print(colored("\tnot saved", "red"))

def first_filter(im_original):
	"""
	Set to black all the pixels lower than 128
	Set to white all the pixels higher than 128
	"""
	new = im_original.copy()
	new_pix = new.load()
	for i in range(0, im_original.size[0]):
		for j in range(0, im_original.size[1]):
			if (new_pix[i,j] > 128):
				new.putpixel((i,j), w1)
	return new

def filterBlock(im_original):
	"""
	Keep "coloured" a sub-matrix 2x2 only if all pixels in it are not white
	"""
	size = (200,60)
	new = Image.new(mode='L',size=size,color=w1)
	pix_original = im_original.load()
	
	for i in range(0, im_original.size[0]-1):
		for j in range(0, im_original.size[1]-1):
			if (pix_original[i,j] != w1 
				and pix_original[i,j+1] != w1 
				and pix_original[i+1,j] != w1 
				and pix_original[i+1,j+1] != w1):

					new.putpixel((i, j), n1)
					new.putpixel((i, j+1), n1)
				
					new.putpixel((i+1, j), n1)
					new.putpixel((i+1, j+1), n1)
	return new

def filterGray(im_original):
	"""
	Keep only the pixels darker then `dark_gray`
	"""
	dark_grey = 0x55

	im = im_original.copy()
	pix = im.load()

	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			if pix[i, j] <= dark_grey:
				im.putpixel((i,j), n1)
			else:
				im.putpixel((i,j), w1)

	return im

def clean_image(im_original):
	bw_img = first_filter(im_original)

	im2 = filterBlock(bw_img)
	im = filterGray(bw_img)

	pix = im.load()
	pix2 = im2.load()

	for i in range(0, im_original.size[0]):
		for j in range(0, im_original.size[1]):
			r = pix[i,j]
			r2 = pix2[i,j]
			pix[i,j] = (r & r2)
	return im.convert('L')

def createDirectories():
	try:
		os.mkdir('./{}/'.format(train_set_folder))
	except:
		return
	for c in string.ascii_uppercase:
		try:
			os.mkdir('./{}/{}/'.format(train_set_folder,c))
		except:
			continue

	
modelChoosen = False
createDirectories()

print("You can use the help of a trained model to speed up the gathering")
try:
	name = input("Input the name of the model to use (if missing than run 'create_model.py' after this program and now press '\\n'): ")
except KeyboardInterrupt:
	print("\nclosing...")
	exit()
except EOFError:
	print("\nclosing...")
	exit()

#name='test-05-9724.h5'
if name != "":
	print("Loading model: "+colored(name, 'blue'))
	try:
		model = load_model(name)
	except OSError:
		print('\n' + colored('Model Not Found!', 'red'))
		exit()
	modelChoosen = True

headers = {
	'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
	'Accept-Encoding': 'gzip, deflate, br',
}

corrects = 0

for k in range(0, 100):
	try:
		r = requests.get('https://forms.iit.it/captcha.php', headers=headers)
	except:
		print(colored("Connection Error, check your internet connection", "red"))
		exit(0)
	
	if r.status_code == 200:
		with open('res.bmp'.format(k), 'wb') as f:
			f.write(r.content)
	else:
		print(colored("Error "+str(r.status_code),"red"))
		continue

	im = clean_image(Image.open('res.bmp').convert('L'))
	im.save('result.bmp')

	tmp = [None]*5
	s=""
	for n in range(5):
		tmp[n] = im.crop( (n*40+10, 18, (n*40)+40, im.size[1]-8) )
	
	if (k < 10): x = '0'+str(k)
	else: x = str(k)

	print('\n' + x + '  ', end='')

	if modelChoosen:
		for t in tmp:
			X_test = np.array(t).reshape(1,34,30,1)
			X_test = X_test.astype('float32')
			X_test /= 255

			results = model.predict(X_test)
			m=0
			r = results[0]

			# get the most probable result
			for i in range(1, len(r)):
				if r[i] > r[m]:
					m = i

			s += chr( ord('A') + m )

		print(colored("AI result: ", 'blue') + s)
		print("- if the result is correct press '\\n' to continue or enter '-' to save the letters and continue")

	try:
		cc = input('enter the right result: ')
	except KeyboardInterrupt:
		terminate_program(corrects,k,modelChoosen)
	except EOFError:
		terminate_program(corrects,k,modelChoosen)

	if cc=='-':
		save_letters(tmp, s)
	
	if len(cc)<=4:
		if modelChoosen:
			print(colored("\tCounted as correct", "green"))
			corrects += 1
		continue

	save_letters(tmp, cc)

	if modelChoosen:
		print(colored("\tCounted as wrong", "red"))

terminate_program(corrects,k+1, modelChoosen)
