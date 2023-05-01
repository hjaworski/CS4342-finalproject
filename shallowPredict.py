import numpy as np
import os
import PIL
import PIL.Image
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 32
IMG_WIDTH = 500
IMG_HEIGHT = 334
IMG_CHANNELS = 3

directory = "train"


def load_image(filename, label):
    image_string = tf.io.read_file(directory + "/" + filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize(image, [50, 34])

    return image, label


def make_tf_dataset(n=1000):
    filenames = sorted(os.listdir(directory))[:n]

    labels = load_labels(len(filenames))

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(load_image, num_parallel_calls=4)

    dataset = dataset.batch(64)

    return dataset


def load_labels(N):
    labels_csv = pd.read_csv("/content/gdrive/MyDrive/Machine Learning Final Project/one_hot_encode_labels.csv")

    array = np.ndarray((N, 5))

    array[:, 0] = labels_csv["scab"][:N]
    array[:, 1] = labels_csv["frog_eye_leaf_spot"][:N]
    array[:, 2] = labels_csv["complex"][:N]
    array[:, 3] = labels_csv["powdery_mildew"][:N]
    array[:, 4] = labels_csv["rust"][:N]

    return array

d = make_tf_dataset(18632)

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(50, 34, IMG_CHANNELS)),
  tf.keras.layers.Dense(5, activation="sigmoid")
])

def TotalAccuracy(y_true, y_pred):
  real_pred = tf.cast(y_pred > 0.5, tf.float32)
  equal = tf.cast(tf.equal(real_pred, y_true), tf.float32)

  sum_equal = tf.math.reduce_sum(equal, axis=1)
  return tf.math.reduce_mean(tf.cast(sum_equal > 4.5, tf.float32))


model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5), TotalAccuracy])

model.fit(d, epochs=5)