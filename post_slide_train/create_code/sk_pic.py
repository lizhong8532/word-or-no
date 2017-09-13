
# coding: utf-8

# In[29]:


import numpy as np
from PIL import Image
import random
import math


# In[30]:


def add_line(array, pointx, pointy,orient):
    if (len(array.shape)>2):
        row, col, channel = array.shape
    else :
        row, col = array.shape
    x = np.arange(0,col)
    y = ((x-pointx)*math.sin(orient)+pointy).astype('int32')
    index_y = np.where((y>=0) & (y<row))
    d=np.asarray(index_y).reshape(1, -1)
    valid_x = x[d]
    valid_y = y[d]
    if (len(array.shape)>2):
        array[valid_y, valid_x, :]=0
    else :
        array[valid_y, valid_x]=0
    return array


# In[31]:


def create_pic(pic_path, pic_name):
    pic_raw = Image.open(pic_path)
    ar = np.array(pic_raw)
    if (len(ar.shape)>2):
        row, col, channel = ar.shape
    else :
        row, col = ar.shape         
    pointy = random.randint(0, row-1)
    pointx = random.randint(0, col-1)
    orient = random.uniform(0, math.pi*2)
    if (row+col<=32): #the pic's scale is too small
        ar = add_line(ar, pointx, pointy, orient)
        ar = add_line(ar, pointx, pointy+1, orient)
    else:
        ar = add_line(ar, pointx, pointy, orient)
        ar = add_line(ar, pointx, pointy+1, orient)
        ar = add_line(ar, pointx, pointy-1, orient)
        ar = add_line(ar, pointx, pointy+2, orient)
    re_im = Image.fromarray(ar)
    re_im.save(pic_name, 'png')

