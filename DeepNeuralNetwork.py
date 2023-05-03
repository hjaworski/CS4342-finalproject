import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load the data
traindata = pd.read_csv('train.csv')

# Make binary problem
traindata['Pawpularity'] = traindata['Pawpularity'].apply(lambda x: 1 if x >= 50 else 0)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(traindata.drop('Pawpularity', axis=1),
                                                    traindata['Pawpularity'],
                                                    test_size=0.2,
                                                    random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = X_train.drop(columns=["Id"])
X_test = X_test.drop(columns=["Id"])
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)
