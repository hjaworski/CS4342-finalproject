# CS4342 Final Project
# Hannah Jaworski, Nathan Kumar, Grace Casey, Casey Snow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

traindata = pd.read_csv('train[1].csv')
testdata = pd.read_csv('test (1).csv')

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
# Generate random predictions for the validation set
y_pred_val = np.random.randint(2, size=len(y_labels))

# Calculate the accuracy of the random baseline model
accuracy = (y_pred_val == y_labels).mean()
print("Accuracy of random baseline model: {:.2%}".format(accuracy))

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

#softmax
# Create a softmax regression model
softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Fit the model to the data
softmax_reg.fit(train_df_scaled, y_train)

# Make predictions on the test data
test_df = X_label.drop(columns=["Id"])
test_df_scaled = StandardScaler().fit_transform(test_df)
y_pred = softmax_reg.predict(test_df_scaled)

# Evaluate the accuracy of the model
acc = accuracy_score(y_labels, y_pred)
print(" soft Accuracy:", acc)
