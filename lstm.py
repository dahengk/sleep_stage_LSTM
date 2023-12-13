# LSTM Model 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.lite.python import lite
import tensorflow as tf


# Load the final combined CSV file
final_combined_file = './final_combined.csv'
data = pd.read_csv(final_combined_file)

# Drop rows with missing values
data = data.dropna()

# Treat -1 as 0
data['sleep_motion'] = data['sleep_motion'].replace(-1, 0) 

# Extract features and labels
features = data.drop(['sleep_motion'], axis=1)  # Drop the label column
labels = data['sleep_motion']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for LSTM model (assuming your data is time series)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(6, activation='softmax'))  # Assuming 6 sleep stages (from 0 to 5)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy[1] * 100:.2f}%")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('sleep_model.tflite', 'wb') as f:
    f.write(tflite_model)
