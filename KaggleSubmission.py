# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import PIL
import PIL.Image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, InputLayer


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def TotalAccuracy(y_true, y_pred):
    real_pred = tf.cast(y_pred > 0.5, tf.float32)
    equal = tf.cast(tf.equal(real_pred, y_true), tf.float32)

    sum_equal = tf.math.reduce_sum(equal, axis=1)
    return tf.math.reduce_mean(tf.cast(sum_equal > 4.5, tf.float32))


def loadData(folder):
    filenames = sorted(os.listdir(folder))
    N = len(filenames)

    IMG_WIDTH = 100
    IMG_HEIGHT = 68

    data = np.ndarray((N, IMG_HEIGHT, IMG_WIDTH, 3))

    for idx, filename in enumerate(filenames):
        image = PIL.Image.open(folder + "/" + filename)
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        data[idx] = image

    return data


def translatePredictions(predictions):
    size = predictions.shape[0]
    translation = [""] * size
    for x in range(size):
        for y in range(5):
            currentPredictor = predictions[x, y]
            if translation[x] != "" and currentPredictor == 1:
                if y == 0:
                    translation[x] = " scab"
                elif y == 1:
                    translation[x] = " frog_eye_leaf_spot"
                elif y == 2:
                    translation[x] = " complex"
                elif y == 3:
                    translation[x] = " powdery_mildew"
                elif y == 4:
                    translation[x] = " rust"
            elif translation[x] == "" and currentPredictor == 1:
                if y == 0:
                    translation[x] = "scab"
                elif y == 1:
                    translation[x] = "frog_eye_leaf_spot"
                elif y == 2:
                    translation[x] = "complex"
                elif y == 3:
                    translation[x] = "powdery_mildew"
                elif y == 4:
                    translation[x] = "rust"

    for string in translation:
        if string == "":
            string = "healthy"

    return translation


def buildSubmissionCSV(directoryTe, finalPredictions):
    filenames = sorted(os.listdir(directoryTe))
    N = len(filenames)

    listFileNames = [""] * N
    for idx, filename in enumerate(filenames):
        listFileNames[idx] = filename

    listFileNames = np.array(listFileNames).T
    finalPredictions = np.array(finalPredictions).T
    submission = np.vstack((listFileNames, finalPredictions)).T
    print(submission.shape)
    pd.DataFrame(submission).to_csv("submission.csv", index=False, index_label=False, header=["image", "labels"])
    return submission

