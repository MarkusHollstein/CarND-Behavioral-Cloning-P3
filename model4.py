import csv
from scipy import ndimage
import numpy as np

lines = []
#with open('../../../opt/carnd_p3/data/driving_log.csv')
with open('data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)

    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)




import cv2
import numpy as np
import sklearn
import os
import csv
import random

def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                image_flipped = np.fliplr(center_image)
                meas_flipped = -center_angle
                images.append(image_flipped)
                angles.append(meas_flipped)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

ch, row, col = 3, 160, 320  # Trimmed image format
from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda,Dropout,BatchNormalization, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Cropping the image to ignore irrelevant areas and make the training faster
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 -0.5, input_shape=(160,320,3)))
model.add(Conv2D(24,(5,5),padding = 'valid',strides = 2,activation = 'relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv2D(36,(5,5),padding = 'valid',strides = 2, activation = 'relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv2D(48,(5,5),padding = 'valid',strides = 2,activation= 'relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv2D(48,(3,3),padding = 'valid',strides = 2,activation = 'relu'))

#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3),padding = 'valid',strides = 2,activation = 'relu'))
model.add(Flatten())
#model.add(Dense(600,activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(100,activation = 'relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(50,activation = 'relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(10,activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(1))


model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator,\
            nb_val_samples=len(validation_samples), nb_epoch=1)
model.save('model4.h5')