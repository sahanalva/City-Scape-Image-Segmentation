import cv2
import os
import random
import numpy as np
from tensorflow.keras.utils import to_categorical

def data_gen(img_folder, mask_folder, batch_size, num_classes = 2,input_shape = (256, 256, 1)):
  c = 0
  n = os.listdir(img_folder) #List of training images
  random.shuffle(n)
  image_width = input_shape[0]
  image_height = input_shape[1]
  image_channels = input_shape[2]
  
  while (True):
    img = np.zeros((batch_size, image_width, image_height, image_channels)).astype('float')
    mask = np.zeros((batch_size, image_width, image_height, num_classes)).astype('float')

    for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 

      train_img = cv2.imread(img_folder+'/'+n[i])/255.
      
      img[i-c] = train_img #add to array - img[0], img[1], and so on.
                                                   

      train_mask = cv2.imread(mask_folder+'/'+n[i].replace('.png', '_trainId_label.png'), -1)
      train_mask = to_categorical(train_mask)

      mask[i-c] = train_mask

    c+=batch_size
    if(c+batch_size>=len(os.listdir(img_folder))):
      c=0
      random.shuffle(n)
                  # print "randomizing again"
    yield img, mask