pip install tensorflow

pip install keras

pip install opencv-python

pip install matplotlib

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

img=image.load_img("dataset/train/biodegradable/TEST_BIODEG_HFL_0.jpg")

plt.imshow(img)

cv2.imread("dataset/train/biodegradable/TEST_BIODEG_HFL_0.jpg")

cv2.imread("dataset/train/biodegradable/TEST_BIODEG_HFL_0.jpg").shape

train= ImageDataGenerator(rescale=1/255)
validation= ImageDataGenerator(rescale= 1/255)

train_dataset= train.flow_from_directory("dataset/train/", target_size=(200,200), batch_size=10, class_mode="binary")
valid_dataset= train.flow_from_directory("dataset/valid/", target_size=(200,200), batch_size=10, class_mode="binary")

train_dataset.class_indices
valid_dataset.class_indices

train_dataset.classes

model= tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape=(200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Flatten(),
    #
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(loss='binary_crossentropy',
             optimizer=RMSprop(lr=0.001),
             metrics=['accuracy'])


model_fit=model.fit(train_dataset,
                    steps_per_epoch=15,
                    epochs=100,
                    validation_data=valid_dataset
                   )

dir_path="dataset/test"
for i in os.listdir(dir_path):
    img=image.load_img(dir_path+'//'+i, target_size=(200,200,3))
    plt.imshow(img)
    plt.show()
    
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    
    result= model.predict(images)
    if result==0:
        print("BIODEGRADABLE")
    else:
        print("NON BIODEGRADABLE")
