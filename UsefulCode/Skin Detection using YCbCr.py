#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import numpy as np
min_YCbCr = np.array([0, 133, 77], np.uint8)
max_YCbCr = np.array([255, 173, 127], np.uint8)


# In[9]:


def detect_skin(image):
    ycbcr_img = cv2. cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycbcr_img, min_YCbCr, max_YCbCr)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin, skin_mask


# In[37]:


def get_skin_colour(image):
    skin, skin_mask = detect_skin(image)
    
    # Calculate the avg color 
    skin_pxls = image[skin_mask > 0] # Get only skin pixels
    if len(skin_pxls) > 0:
        avg_colour = np.mean(skin_pxls, axis = 0)
    else:
        avg_colour = [0,0,0]  #Default if no skin is detected
    return avg_colour


# In[22]:


def draw_skin_colour(image, skin_colour):
    colour = (int(skin_colour[0]), int(skin_colour[1]), int(skin_colour[2]))
    cv2.putText(image, "Skin Colour: ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.rectangle(image, (150, 10), (180, 40), colour, -1)


# In[32]:


def increase_brightness(colour, factor = 1.675):
    return tuple(min(int(c * factor), 255) for c in colour)


# In[40]:


video = cv2.VideoCapture(0)

while True:
    (check, image) = video.read()
    if not check:
        break
        
    skin, skin_mask = detect_skin(image)
    avg_colour = get_skin_colour(image)
    brighten_colour = increase_brightness(avg_colour)
    draw_skin_colour(image, brighten_colour)
    
    cv2.imshow("Original Image", image)
    cv2.imshow("Skin Detected", skin)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()


# In[39]:


image1 = cv2.VideoCapture(0)
(check, image1) = image1.read()
avg_colour = get_skin_colour(image1)
brighten_colour = increase_brightness(avg_colour)
draw_skin_colour(image1, brighten_colour)
cv2.imshow("Captured Image", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




