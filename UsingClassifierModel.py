# importing the libraries
import keras
from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

# intialize the class labels for the cifar10 dataset
class_labels = [
    "cat",
    "dog",
    ]

# load the json file that contain neural structure
f = Path("classifier_structure.json")
model_structure = f.read_text()

# recreate the model from the json file
model = model_from_json(model_structure)

#reload the weights to the neurons
model.load_weights("model_weights.h5")

# load the image file
img = image.load_img("dog.jpg", target_size=(64,64))

# convert to the image array
img_to_test = image.img_to_array(img)
img_to_test=np.expand_dims(img_to_test, axis = 0)
result = model.predict(img_to_test)

# checking the prediction Result
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
# Print the result
print("This image is a {}",prediction)

