import os
import numpy as np
from tensorflow.keras.preprocessing import image
import keras.preprocessing.image

#Quick method to get the most likely classification of any given image.
#Takes in model and image path for arguments
def test(model,img_path):

  img = process_img_path(img_path)
  x = keras.preprocessing.image.img_to_array(img)
  result = model.predict(x)
  #To generalize we don't have a decode predictions method, however first line should be self explanatory
  print(result)
