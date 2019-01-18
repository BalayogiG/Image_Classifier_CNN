# importing packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Initializing the CNN
classifier = Sequential()

# step 1 : Convolution
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))

# step 2 : Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

# adding second convolution layer
classifier.add(Conv2D(32, (3,3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

# step 3 : Flattening
classifier.add(Flatten())

# step 4 : Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# part 2 - Filtering the CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('training_set', target_size=(64,64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('test_set', target_size=(64,64), batch_size=32, class_mode='binary')
classifier.fit_generator(training_set, steps_per_epoch=100, epochs=5, validation_data=test_set, validation_steps=2000)

#saving the neural network structure
classifier_model_structure= classifier.to_json()
f = Path("classifier_structure.json")
f.write_text(classifier_model_structure)

# saving the trained weights
classifier.save_weights("model_weights.h5")

