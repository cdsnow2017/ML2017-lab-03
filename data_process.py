# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:05:21 2017

@author: cdsnow
"""

import numpy as np
from PIL import Image
from feature import NPDFeature
import pickle
import os

def get_npdArray_from_diskImg(pathDir):
    list = []
    file_names =  os.listdir(pathDir)
    for file_name in file_names:
        img = Image.open('%s%s' % (pathDir, file_name))
        img = img.resize((24, 24))
        img = np.array(img.convert('L'))
        npdFeature = NPDFeature(img)
        npdArray = npdFeature.extract()
        list.append(npdArray)
    faces_npdArray = np.array(list)
    return faces_npdArray
    
def save(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

#f=open('dump/NPD_noface','rb')
#disk_npd = pickle.load(f)