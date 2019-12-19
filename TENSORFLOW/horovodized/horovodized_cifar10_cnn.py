
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from horovodizer_helper_TF import *
import horovod.keras as hvd
import math
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import backend as K
from tensorflow.keras.optimizers import get as get_optimizer_by_name
hvd.init()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))
batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(Conv2D(32, (3, 3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Conv2D(64, (3, 3), padding='same'))
model2.add(Activation('relu'))
model2.add(Conv2D(64, (3, 3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(512))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes))
model2.add(Activation('softmax'))
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-06)
opt2 = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=adapt_optimizer(opt2), metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy', optimizer=adapt_optimizer(opt2), metrics=['accuracy'])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
if (not data_augmentation):
    print('Not using data augmentation.')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=adapt_epochs(epochs), validation_data=(x_test, y_test), shuffle=True, callbacks=adapt_callbacks([], True), verbose=(1 if (hvd.rank() == 0) else 0))
    model2.fit(x_train, y_train, batch_size=batch_size, epochs=adapt_epochs(epochs), validation_data=(x_test, y_test), shuffle=True, callbacks=adapt_callbacks([], True), verbose=(1 if (hvd.rank() == 0) else 0))
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=True, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0)
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_test, y_test), workers=4, callbacks=adapt_callbacks([], True), verbose=(1 if (hvd.rank() == 0) else 0))
if (not os.path.isdir(save_dir)):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
if (hvd.rank() == 0):
    model.save(model_path)
print(('Saved trained model at %s ' % model_path))
if (hvd.rank() == 0):
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
