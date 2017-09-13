
# coding: utf-8

# In[4]:


import os
import sk_pic


# In[5]:


def convert_all(root_path, to_path, times):
    for i in range(times): #造20次
        for _, _, img_list in os.walk(root_path):
            for img_name in img_list:
                name, ext = img_name.split('.', 1)
                if (ext == "png" or ext == 'jpg'):
                    pic_path = root_path + img_name
                    pic_name = to_path + name + str(i) + '.' + ext
                    sk_pic.create_pic(pic_path, pic_name)
                    


# In[ ]:





# In[ ]:




