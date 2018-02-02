import csv
import cv2
import numpy as np
from keras.layers import Input, Dense, Flatten, Lambda, Activation
from keras.models import Model, Sequential
from keras.layers.convolutional import Convolution2D

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_lines, validation_lines = train_test_split(lines[1:], test_size=0.3) # By default: data shuffled before splitting 

from sklearn.utils import shuffle

def generator(lines, batch_size=32):
    num_lines = len(lines)
    while True: # Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]
    
            images = []
            measurements = []
            for batch_line in batch_lines:
                source_path = batch_line[0]
                filename = source_path.split('/')[-1]
                current_path = 'data/IMG/' + filename
                image = cv2.imread(current_path)
                images.append(image)
                measurement = float(batch_line[3])
                measurements.append(measurement)
            
            # trim images to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield(shuffle(X_train, y_train))

# the generators
train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)


# define model
#input_shape = X_train.shape[1:]
#inp = Input(shape=input_shape)
#x = Flatten()(inp)
#x = Dense(1)(x)
#model = Model(inp, x)

row, col, ch = 160, 320, 3

# Define Model
model = Sequential()
# 1. normalization layer
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
# 2. convolutional layer
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2)))
model.add(Activation('relu'))
# 3. convolutional layer
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2)))
model.add(Activation('relu'))
# 4. convolutional layer
model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2)))
model.add(Activation('relu'))
# 5. convolutional layer
model.add(Convolution2D(64, 5, 5, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
# 6. convolutional layer
model.add(Convolution2D(64, 5, 5, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
model.add(Flatten())
# 7. fully-connected layer
model.add(Dense(100))
model.add(Activation('relu'))
# 8. fully-connected layer
model.add(Dense(50))
model.add(Activation('relu'))
# 9. fully-connected layer
model.add(Dense(10))
model.add(Activation('relu'))
# output
model.add(Dense(1))

# Train Model
print("Training...")
model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, \
                    samples_per_epoch=len(train_lines), \
                    validation_data=validation_generator, \
                    nb_val_samples=len(validation_lines), \
                    nb_epoch=10)


# Save Model
model.save('model.h5')
print("Model saved.")
