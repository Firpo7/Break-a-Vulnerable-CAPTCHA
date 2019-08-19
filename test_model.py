#!/usr/bin/env python3

from keras.models import load_model
from termcolor import colored
from keras.utils import to_categorical
from utils import *
import string, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_confusion_matrix(cm, title='Model precision', cmap='gray', labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap='gray')
    plt.title(title)
    fig.colorbar(cax)
    if labels:
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
    plt.xlabel('Letter  recognised')
    plt.ylabel('Letter to recognise')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ab = string.ascii_uppercase

print("Using this program you can test the accuracy of one of your models and see where it fails")

try:
	name = input("Input the name of the model to test: ")
except KeyboardInterrupt:
	print("\nclosing...")
	exit()
except EOFError:
	print("\nclosing...")
	exit()

if name != '':
	print("Loading model: "+colored(name, 'blue'))
	try:
		model = load_model(name)
		print("please wait...")
	except OSError:
		print('\n' + colored('Model Not Found!', 'red'))
		exit()
else:
	exit()

(X_test, y_test, tot_test) = load_data(useTest = True)

X_test = X_test.reshape(tot_test,34,30,1)
X_test = X_test.astype('float32')
X_test /= 255

results = model.predict(X_test)

stat = {}
tot = {}

for c in ab:
	stat[c]=0
	tot[c]=0

k=0
M = np.zeros((26,26))
for r in results:
	n=0
	
	# get the most probable for each result
	for i in range(1, len(r)):
		if r[i] > r[n]:
			n = i
	
	l = chr(y_test[k]+ord('A'))
	tot[l]+=1

	M[y_test[k], n] += 1
	
	if n == y_test[k]:
		stat[l]+=1
		
	k+=1

Mnorm = np.zeros((26,26))
m=0
for i in range(26):
    n=0
    for j in range(26):
        Mnorm[m, n] = M[m,n]/tot[ab[i]]
        n+=1
    m+=1

print()
corrects = 0
for c in ab:
	print(c+':\t'+str(stat[c])+'/'+str(tot[c]))
	corrects+=stat[c]

print('\nTotal:\t'+str(corrects)+'/'+str(tot_test), end='\n\n')

y_test = to_categorical(y_test)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1], end='\n\n')

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
plot_confusion_matrix(Mnorm, labels=labels)
