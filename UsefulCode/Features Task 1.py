#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui


# In[30]:


camera = cv2.VideoCapture(0)
(grabbed, I) = camera.read()
plt.imshow(I)


# In[31]:


gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')


# In[32]:


##Gradients from a gray scale image
Ix = cv2.Sobel(gray,ddepth=cv2.CV_64F,dx=1,dy=0)
Iy = cv2.Sobel(gray,ddepth=cv2.CV_64F,dx=0,dy=1)


# In[33]:


##Edges from original image
E = cv2.Canny(I,threshold1=100,threshold2=200)


# In[34]:


##Find the gradient magnitude
Gm = np.sqrt(Ix**2 + Iy**2)
Gm = cv2.normalize(Gm, None, 0, 255, cv2.NORM_MINMAX)


# In[35]:


##Overlay gm on edges
Gm_edges = np.zeros_like(Gm)
Gm_edges[E > 0] = Gm[E > 0]


# In[36]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Edge Mask (Canny)")
plt.imshow(E, cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.title("Gradient Magnitude on Edges")
plt.imshow(Gm_edges, cmap='gray')
plt.axis('off')
plt.show()


# In[37]:


cv2.imshow("Edge Mask", Gm_edges)
cv2.imshow("Gradient Magnitude", Gm_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

