#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import cv


# In[25]:


import cv2


# In[26]:


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")


# In[27]:


with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#different colours at time of object detection
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# In[28]:


print(classes)


# In[29]:


#loding blob image into the network i.e algorithm
img = cv2.imread("room_ser.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape


# In[30]:


#blob is the way to extract details from the images

#altering somechanges so as to 
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)


# In[31]:


# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            
              #to get coreness of the object
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
             # that needed to be referred as when we detect images
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
             #to know the name of id's we detected
            class_ids.append(class_id)


# In[32]:



indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#you need to take indexes which are not overlapping i.e two rectangles only for one object
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        #to display the detection rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
          #to display the label
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        


# In[33]:



cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




