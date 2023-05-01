import pandas
import numpy as np
import homework3_cesnow2 as hw3


def make_matrix(focus, action, face, group):
    mat = np.vstack((focus, action))
    mat = np.vstack((mat, face))
    mat = np.vstack((mat, group))
    mat = np.vstack((mat, np.ones(mat.shape[1])))
    return mat.T


if __name__ == "__main__":
    # Load training data
    d = pandas.read_csv("train.csv")
    y = d.Pawpularity.to_numpy()
    y_onehot = np.zeros((y.shape[0], 2))
    y_onehot[np.arange(y.shape[0]), y] = 1
    y = y_onehot
    focus = d.Pclass.to_numpy()
    action = d.Pclass.to_numpy()
    face = d.Pclass.to_numpy()
    group = d.Pclass.to_numpy()

    x = make_matrix(focus, action, face, group)

    # Train model using part of homework 3.
    # ...
    w = hw3.softmaxRegression(x, y, epsilon=0.1, batchSize=10, num_classes=2, alpha=.1)
    # Load testing data
    # ...
    d_test = pandas.read_csv("test.csv")
    photo_ids = d_test.id.to_numpy()
    focus_test = d_test.SubjectFocus.to_numpy()
    action_test = d_test.Action.to_numpy()
    face_test = d_test.Face.to_numpy()
    group_test = d_test.Group.to_numpy()
    x_test = make_matrix(focus_test, action_test, face_test, group_test)
    # Compute predictions on test set
    # ...
    num_examples = x_test.shape[0]
    scores = x_test.dot(w)
    yhat_test = hw3.softmax(scores)
    yhat_test = np.argmax(yhat_test, axis=1)
    yhat_test = np.vstack((photo_ids, yhat_test))
    yhat_test = yhat_test.T
    # Write CSV file of the format:
    # ..., ...
    print(yhat_test)
    np.savetxt('submission.csv', yhat_test, fmt='%d', delimiter=",")
