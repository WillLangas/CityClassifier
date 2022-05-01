import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.densenet import preprocess_input

from keras.models import model_from_json
import os

global prediction
global top3

labels = ["Amsterdam", "Austin", "Boston", "Budapest", "Helsinki", "London", "Manila", "Melbourne", "Miami", "Phoenix", "Sanfrancisco", "Saopaolo", "Tokyo", "Toronto", "Trondheim", "Zurich"]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

def PredictImage(imgPath):
    global prediction
    global top3
    image = load_img(imgPath, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    results = loaded_model.predict(image)
    pred = np.argmax(results)
    prediction = labels[pred]
    #print(prediction)
    top_values_index = sorted(range(len(results[0])), key=lambda i: results[0][i])[-3:]
    top3ind = top_values_index[::-1]
    top3 = [labels[top3ind[0]], labels[top3ind[1]], labels[top3ind[2]]]
    #print(top3)
    return prediction, top3