
# coding: utf-8

# In[15]:


import os
import numpy as np
from PIL import Image
import random


# In[16]:


def create_right_background(root_path, background_path, to_path, str1):
    back = Image.open(background_path)
    width1, height1 = back.size
    for _, _, imglist in os.walk(root_path):
        for img_name in imglist:
            name, ext = img_name.split('.', 1)
            if (ext=='jpg' or ext=='png'):
                imgpath = root_path + img_name
                front = Image.open(imgpath)
                new = Image.new('RGB', front.size, (255,255,255)) # create a new blank pic
                width, height = front.size
                if (height1>height):
                    point = random.randint(0, height1-height)
                    bound1 = width//3
                    box = (0, point, bound1, point+height)
                    region = back.crop(box)
                    box1=(width-bound1, 0, width, height)
                    new.paste(region,box1)
                    frontarr = np.array(front)
                    newarr = np.array(new)
                    changearr = np.where(frontarr<=180, frontarr, newarr)
                    re_im = Image.fromarray(changearr)
                    re_name = to_path + name + '_' + str1 + '_' + str(1) + '.' +ext
                    re_im.save(re_name)

def create_left_background(root_path, background_path, to_path, str1):
    back = Image.open(background_path)
    width1, height1 = back.size
    for _, _, imglist in os.walk(root_path):
        for img_name in imglist:
            name, ext = img_name.split('.', 1)
            if (ext=='jpg' or ext=='png'):
                imgpath = root_path + img_name
                front = Image.open(imgpath)
                new = Image.new('RGB', front.size, (255,255,255)) # create a new blank pic
                width, height = front.size
                if (height1>height):
                    point = random.randint(0, height1-height)
                    bound1 = width//3
                    box = (width1-bound1, point, width1, point+height)
                    region = back.crop(box)
                    box1=(0, 0, bound1, height)
                    new.paste(region,box1)
                    frontarr = np.array(front)
                    newarr = np.array(new)
                    changearr = np.where(frontarr<=180, frontarr, newarr)
                    re_im = Image.fromarray(changearr)
                    re_name = to_path + name + '_' + str1 + '_' + str(2) + '.' +ext
                    re_im.save(re_name)
                    
def create_up_background(root_path, background_path, to_path, str1):
    back = Image.open(background_path)
    width1, height1 = back.size
    for _, _, imglist in os.walk(root_path):
        for img_name in imglist:
            name, ext = img_name.split('.', 1)
            if (ext=='jpg' or ext=='png'):
                imgpath = root_path + img_name
                front = Image.open(imgpath)
                new = Image.new('RGB', front.size, (255,255,255)) # create a new blank pic
                width, height = front.size
                if (width1>width):
                    point = random.randint(0, width1-width)
                    bound1 = height//3
                    box = (point, height1-bound1, point+width, height1)
                    region = back.crop(box)
                    box1=(0, 0, width, bound1)
                    new.paste(region,box1)
                    frontarr = np.array(front)
                    newarr = np.array(new)
                    changearr = np.where(frontarr<=180, frontarr, newarr)
                    re_im = Image.fromarray(changearr)
                    re_name = to_path + name + '_' + str1 + '_' + str(3) + '.' +ext
                    re_im.save(re_name)
                    
def create_down_background(root_path, background_path, to_path, str1):
    back = Image.open(background_path)
    width1, height1 = back.size
    for _, _, imglist in os.walk(root_path):
        for img_name in imglist:
            name, ext = img_name.split('.', 1)
            if (ext=='jpg' or ext=='png'):
                imgpath = root_path + img_name
                front = Image.open(imgpath)
                new = Image.new('RGB', front.size, (255,255,255)) # create a new blank pic
                width, height = front.size
                if (width1>width):
                    point = random.randint(0, width1-width)
                    bound1 = height//3
                    box = (point, 0, point+width, bound1)
                    region = back.crop(box)
                    box1=(0, height-bound1, width, height)
                    new.paste(region,box1)
                    frontarr = np.array(front)
                    newarr = np.array(new)
                    changearr = np.where(frontarr<=180, frontarr, newarr)
                    re_im = Image.fromarray(changearr)
                    re_name = to_path + name + '_' + str1 + '_' + str(4) + '.' +ext
                    re_im.save(re_name)


# In[17]:


front_path = '/home/com/Documents/createpic/train/text/'
back_path = '/home/com/Documents/iconLib/'
to_path = '/home/com/Documents/text_add_pic/'

for _, _, imglist in os.walk(back_path):
    for img_name in imglist:
        name, ext = img_name.split('.', 1)
        if (ext=='jpg' or ext=='png'):
            imgpath = back_path + img_name
            create_left_background(front_path, imgpath, to_path, name)
            create_right_background(front_path, imgpath, to_path, name)
            create_up_background(front_path, imgpath, to_path, name)
            create_down_background(front_path, imgpath, to_path, name)


# In[ ]:




