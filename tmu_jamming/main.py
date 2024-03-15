#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import imghdr
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage import io
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
from time import time
import itertools
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pickle

import random
import math
import cv2
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer


# In[5]:


def assign_label(img_type):
    classes = {"11": 0, "12": 1, "31": 2, "32": 3, "34": 4, "40": 5, "chirp_uneven": 6, "not_sweep": 7, "pulse": 8, "sine": 9}
    return classes.get(img_type, -1)  # Return -1 if class is not found

def make_data(img_type, DIR, X_data, y_data):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img_type)
        path = os.path.join(DIR, img)
        
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (50, 50))
        
        X_data.append(np.array(img))
        y_data.append(label)

def create_train_data(dataset_name, dataset_type): 
    X_train = []
    y_train = []
    main_path = os.path.join('/data/data', dataset_name) 

    data_dir = os.path.join(main_path, dataset_type)

    for class_folder in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_folder)
        make_data(class_folder, class_dir, X_train, y_train)

    return np.array(X_train), np.array(y_train)




# In[ ]:


#Spectrograms

X_train_spec, y_train_spec = create_train_data("spectrogram_v2", "train")
X_test_spec, y_test_spec = create_train_data("spectrogram_v2", "test")

y_train = y_train_spec
y_test = y_test_spec

#convert images from colored to gray
X_train_spec = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train_spec])
X_test_spec = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test_spec])

# convert iamges from gray to binary form
for i in range(X_train_spec.shape[0]):
    _, X_train_spec[i,:] = cv2.threshold(X_train_spec[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
for i in range(X_test_spec.shape[0]):
    _, X_test_spec[i,:] = cv2.threshold(X_test_spec[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 


Tm_spec = TMClassifier(
        number_of_clauses=500,
        T=250,
        s=10,
        max_included_literals=32,
        weighted_clauses=True,
        patch_dim=(10, 10)
    )


# In[ ]:


#dominant frequency

X_train_dmof, y_train_dmof = create_train_data("dominant_freq", "train")
X_test_dmof, y_test_dmof = create_train_data("dominant_freq", "test")


#convert images from colored to gray
X_train_dmof = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train_dmof])
X_test_dmof = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test_dmof])

# convert iamges from gray to binary form
for i in range(X_train_dmof.shape[0]):
    _, X_train_dmof[i,:] = cv2.threshold(X_train_dmof[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
for i in range(X_test_dmof.shape[0]):
    _, X_test_dmof[i,:] = cv2.threshold(X_test_dmof[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 


Tm_dmof = TMClassifier(
        number_of_clauses=500,
        T=250,
        s=10,
        max_included_literals=32,
        weighted_clauses=True,
        patch_dim=(10, 10)
    )


# In[ ]:


#psd

X_train_psd, y_train_psd = create_train_data("psd", "train")
X_test_psd, y_test_psd = create_train_data("psd", "test")


#convert images from colored to gray
X_train_psd = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train_psd])
X_test_psd = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test_psd])

# convert iamges from gray to binary form
for i in range(X_train_psd.shape[0]):
    _, X_train_psd[i,:] = cv2.threshold(X_train_psd[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
for i in range(X_test_psd.shape[0]):
    _, X_test_psd[i,:] = cv2.threshold(X_test_psd[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 


Tm_psd = TMClassifier(
        number_of_clauses=500,
        T=250,
        s=10,
        max_included_literals=32,
        weighted_clauses=True,
        patch_dim=(10, 10)
    )


# In[ ]:


#fft

X_train_fft, y_train_fft = create_train_data("fft", "train")
X_test_fft, y_test_fft = create_train_data("fft", "test")


#convert images from colored to gray
X_train_fft = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train_fft])
X_test_fft = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test_fft])

# convert iamges from gray to binary form
for i in range(X_train_fft.shape[0]):
    _, X_train_fft[i,:] = cv2.threshold(X_train_fft[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
for i in range(X_test_fft.shape[0]):
    _, X_test_fft[i,:] = cv2.threshold(X_test_fft[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 


Tm_fft = TMClassifier(
        number_of_clauses=500,
        T=250,
        s=10,
        max_included_literals=32,
        weighted_clauses=True,
        patch_dim=(10, 10)
    )


# In[ ]:


#std

X_train_std, y_train_std = create_train_data("std", "train")
X_test_std, y_test_std = create_train_data("std", "test")


#convert images from colored to gray
X_train_std = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train_std])
X_test_std = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test_std])

# convert iamges from gray to binary form
for i in range(X_train_std.shape[0]):
    _, X_train_std[i,:] = cv2.threshold(X_train_std[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
for i in range(X_test_std.shape[0]):
    _, X_test_std[i,:] = cv2.threshold(X_test_std[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 


Tm_std = TMClassifier(
        number_of_clauses=500,
        T=250,
        s=10,
        max_included_literals=32,
        weighted_clauses=True,
        patch_dim=(10, 10)
    )


# In[ ]:


for epoch in range(100):
    print("#%d" % (epoch+1), end=' ')
    Tm_spec.fit(X_train_spec, y_train)
    y_test_spec, y_test_scores_spec = Tm_spec.predict(X_test_spec, return_class_sums=True)
    print("spectrograms: %.1f%%" % (100*(y_test_spec == y_test).mean()), end=' ')

    Tm_dmof.fit(X_train_dmof, y_train)
    y_test_dmof, y_test_scores_dmof = Tm_dmof.predict(X_test_dmof, return_class_sums=True)
    print("Dominant frequency: %.1f%%" % (100*(y_test_dmof == y_test).mean()), end=' ')

    Tm_psd.fit(X_train_psd, y_train)
    y_test_psd, y_test_scores_psd = Tm_psd.predict(X_test_psd, return_class_sums=True)
    print("Psd: %.1f%%" % (100*(y_test_psd == y_test).mean()), end=' ')

    Tm_std.fit(X_train_std, y_train)
    y_test_std, y_test_scores_std = Tm_std.predict(X_test_std, return_class_sums=True)
    print("Std: %.1f%%" % (100*(y_test_std == y_test).mean()), end=' ')
	
    Tm_fft.fit(X_train_fft, y_train)
    y_test_fft, y_test_scores_fft = Tm_fft.predict(X_test_fft, return_class_sums=True)
    print("fft: %.1f%%" % (100*(y_test_fft == y_test).mean()), end=' ')


# In[ ]:


votes = np.zeros(y_test_scores_spec.shape, dtype=np.float32)
for i in range(y_test.shape[0]):
    votes[i] += 1.0*y_test_scores_dmof[i]/(np.max(y_test_scores_dmof) - np.min(y_test_scores_dmof))
    votes[i] += 1.0*y_test_scores_psd[i]/(np.max(y_test_scores_psd) - np.min(y_test_scores_psd))
    votes[i] += 1.0*y_test_scores_std[i]/(np.max(y_test_scores_std) - np.min(y_test_scores_std))
    votes[i] += 1.0*y_test_scores_std[i]/(np.max(y_test_scores_fft) - np.min(y_test_scores_fft))
    votes[i] += 1.0*y_test_scores_spec[i]/(np.max(y_test_scores_spec) - np.min(y_test_scores_spec))
y_test_team = votes.argmax(axis=1)

print("Team: %.1f%%" % (100*(y_test_team == y_test).mean()))
print()
