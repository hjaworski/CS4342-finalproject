# CS4342 Final Project
# Hannah Jaworski, Nathan Kumar, Grace Casey, Casey Snow
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout

traindata = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')

#make binary problem
traindata['popularity_binary'] = traindata['Pawpularity'].apply(lambda x: 1 if x >= 50 else 0)

# Generate a random array of 1s and 0s for test data.... this should be done on the testing set (that we have the labels for)
random_pawpularity_values = np.random.randint(2, size=len(traindata))

# Add the new column to the testdata
traindata['Random_Baseline_Pawpularity'] = random_pawpularity_values

# Calculate accuracy of the random baseline
accuracy = (traindata['Random_Baseline_Pawpularity'] == traindata['popularity_binary']).mean()

# Print the accuracy
print("Accuracy of random baseline: {:.2%}".format(accuracy))


#traintest split
X_train, X_label, y_train, y_labels = train_test_split(traindata.drop('popularity_binary', axis=1),
                                                  traindata['popularity_binary'],
                                                  test_size=0.2,
                                                  random_state=42)

### PCA visualization
# Scale the data
train_df = X_train.drop(columns=["Id"])
train_df_scaled = StandardScaler().fit_transform(train_df)
# PCA transformation
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(train_df_scaled)
# Plot PCA
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

# softmax
# Create a softmax regression model
softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# y_pred = hw3.softmaxRegression(train_df_scaled, y_train, epsilon=0.1, batchSize=10, num_classes=2, alpha=.1)

# Fit the model to the data
softmax_reg.fit(train_df_scaled, y_train)

# Make predictions on the test data
test_df = X_label.drop(columns=["Id"])
test_df_scaled = StandardScaler().fit_transform(test_df)

y_pred = softmax_reg.predict(test_df_scaled)

# Evaluate the accuracy of the model
acc = accuracy_score(y_labels, y_pred)
print(" soft Accuracy:", acc)


# Neural network
# Scale the data
train_df = X_train.drop(columns=["Id"])
train_df_scaled = StandardScaler().fit_transform(train_df)
test_df = X_label.drop(columns=["Id"])
test_df_scaled = StandardScaler().fit_transform(test_df)

# Create a 3-layer neural network with 1 hidden layer
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# Train the neural network
mlp.fit(train_df_scaled, y_train)

# Make predictions on the validation data
y_pred = mlp.predict(test_df_scaled)

# Evaluate the accuracy of the model
acc = accuracy_score(y_labels, y_pred)
print("Neural network accuracy:", acc)


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.drop('Id', axis=1))
X_label_scaled = scaler.transform(X_label.drop('Id', axis=1))

# Define the model
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_label_scaled, y_labels))

# Evaluate the model
test_df_scaled = scaler.transform(testdata.drop('Id', axis=1))
y_pred = model.predict_classes(test_df_scaled)
print("DNN Accuracy:", accuracy_score(testdata['Pawpularity'].apply(lambda x: 1 if x >= 50 else 0), y_pred))

