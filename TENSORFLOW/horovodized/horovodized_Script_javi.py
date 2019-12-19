
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import IPython.display as display
from IPython.display import Image
from tensorflow.keras.layers import Dense, Input
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from efficientnet.tfkeras import EfficientNetB7, EfficientNetB0
from horovodizer_helper import *
import horovod.tensorflow.keras as hvd
import math
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import backend as K
from tensorflow.keras.optimizers import get as get_optimizer_by_name


def main(_):
    hvd.init()
    print("After hvd init")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    # K.set_session(tf.Session(config=config))
    print("After gpu_options visible_device_list")
    tf.enable_eager_execution(config=config)
    epochs = 20
    steps_per_epoch = 2
    batch_size = 32
    num_classes = 10
    full_model = 'image'
    image_model = 'efficientnet'
    image_training_type = 'finetuning'
    text_model = 'cnn'
    combined_embeddings = 'stack'
    learning_rate = 0.005
    width = 150
    height = 150
    input_shape = (height, width, 3)
    input_size = (224, 224, 3)
    train_tfrecord = tf.data.TFRecordDataset(filenames=['tfrecords/train.tfrecords'])
    print(train_tfrecord)
    val_tfrecord = tf.data.TFRecordDataset(filenames=['tfrecords/val.tfrecords'])
    test_tfrecord = tf.data.TFRecordDataset(filenames=['tfrecords/test.tfrecords'])

    def read_tfrecord(serialized_example):
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)
        input_2 = tf.image.decode_png(example['image_raw'], channels=3, dtype=tf.dtypes.uint8)
        input_2 = tf.image.resize(input_2, [600, 600])
        return (input_2, example['label'])
    train_parsed_dataset = train_tfrecord.map(read_tfrecord)
    val_parsed_dataset = val_tfrecord.map(read_tfrecord)
    test_parsed_dataset = test_tfrecord.map(read_tfrecord)
    tf.keras.backend.clear_session()
    baseModel = EfficientNetB7(weights='imagenet', include_top=True)
    probs = baseModel.layers.pop()
    top_droput = probs.input
    headModel = layers.Dense(10, activation='softmax')(top_droput)
    model = models.Model(inputs=baseModel.input, outputs=headModel)
    SGD = optimizers.SGD(lr=0.01, decay=4e-05, momentum=0.9)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adapt_optimizer(SGD), metrics=['accuracy'])
    train_dataset = train_parsed_dataset.batch(2).repeat()
    val_dataset = val_parsed_dataset.batch(2).repeat()
    test_dataset = test_parsed_dataset.batch(2).repeat()
    model.fit(train_dataset, epochs=adapt_epochs(epochs), steps_per_epoch=400, validation_data=val_dataset, validation_steps=100, verbose=(1 if (hvd.rank() == 0) else 0), callbacks=adapt_callbacks([], True))
    if (hvd.rank() == 0):
        model.save('saved_model.h5')
    if (hvd.rank() == 0):
        (test_loss, test_acc) = model.evaluate(test_dataset, verbose=0, steps=1241)
        print('Test loss =', test_loss)
        print('Test acc =', test_acc)

if __name__ == "__main__":
    tf.app.run()
