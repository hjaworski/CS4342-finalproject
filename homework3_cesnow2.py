import numpy as np
import matplotlib.pyplot as plt


# Softmax regression function
def softmax(X):
    exp_scores = np.exp(X)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# cross-entropy loss function - regularized
def cross_entropy_loss(w, X, y, reg):
    num_examples = X.shape[0]
    scores = X.dot(w)
    yhat = softmax(scores)
    loss = -np.log(yhat[range(num_examples), y])
    data_loss = np.sum(loss) / num_examples
    reg_loss = 0.5 * reg * np.sum(w * w)
    loss = data_loss + reg_loss
    return loss, yhat


# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
def softmaxRegression(trainingImages, trainingLabels, epsilon, batchSize, num_classes, alpha):
    num_features = trainingImages.shape[1]
    num_examples = trainingImages.shape[0]

    # Initialize weights
    W = 0.00001 * np.random.randn(num_features, num_classes)

    num_batches = int(num_examples / batchSize)

    for i in range(500):
        shuffle = np.random.permutation(np.arange(num_examples))
        trainingImages = trainingImages[shuffle]
        trainingLabels = trainingLabels[shuffle]
        start = 0
        end = batchSize

        for j in range(num_batches):
            # mini-batches
            X_batch = trainingImages[start:end]
            y_batch = trainingLabels[start:end]

            loss, yhat = cross_entropy_loss(W, X_batch, np.argmax(y_batch, axis=1), alpha)
            gradient = (1/num_examples) * X_batch.T.dot(yhat - y_batch)

            W = W - epsilon * gradient

            # Print loss
            if i == 499 and j > 579:
                print("Minibatch " + str(j) + " loss = " + str(loss))

        if i% 20 == 0:
            print(i)
        start += batchSize
        end += batchSize

    return W


if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    # ...
    trainingImages = np.vstack((trainingImages.T, np.ones(trainingImages.shape[0])))
    trainingImages = trainingImages.T
    testingImages = np.vstack((testingImages.T, np.ones(testingImages.shape[0])))
    testingImages = testingImages.T

    # Change from 0-9 labels to "one-hot" binary vector labels. For instance,
    # if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    # ...
    onehot_training = np.zeros((trainingLabels.shape[0], 10))
    onehot_training[np.arange(trainingLabels.shape[0]), trainingLabels] = 1

    # Train the model - have to do this with onehot_training
    Wtilde = softmaxRegression(trainingImages, onehot_training, epsilon=0.1,
                               batchSize=100, num_classes=10, alpha=.1)

    # Compute accuracy
    yhat = testingImages.dot(Wtilde)
    predicted_labels = np.argmax(yhat, axis=1)
    accuracy = np.mean(predicted_labels == testingLabels) * 100
    print("Test accuracy: " + str(accuracy) + " %")

    # Visualize the vectors
    # ...
    for i in range(10):
        weight_vector = Wtilde[:-1, i]
        image = weight_vector.reshape(28, 28)
        plt.subplot(2, 5, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title("Title " + str(i))
    plt.show()
