import cv2
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
import time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras import regularizers
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add placeholders for evaluation metrics
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []
confusion_matrices = []

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Final_Data')
os.makedirs(DATA_PATH, exist_ok=True)

# Actions that we try to detect
actions = np.array(['Blue', 'Dance', 'Friend', 'Happy', 'Hello', 'Help', 'Jump', 'Laugh', 'Please', 'Red', 'Run', 'Sit', 'Sorry', 'Stand', 'Stop', 'Thanks', 'Wait', 'Work', 'Deaf', 'Play', 'Family'])
no_sequences = 70
sequence_length = 50

# Label map
label_map = {label: num for num, label in enumerate(actions)}
print(label_map)

sequences, labels = [], []
unexpected_actions = set()  # Use a set to avoid duplicates

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                if res.shape != (1662,):  # Assuming each keypoint array should be of size 1662
                    unexpected_actions.add(action)
                window.append(res)
            except Exception as e:
                print(f"Error loading {action} sequence {sequence} frame {frame_num}: {e}")
        if len(window) == sequence_length:  # Ensure all windows are of the correct length
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Skipped {action} sequence {sequence} due to incorrect length: {len(window)}")

# Output unexpected actions for debugging
if unexpected_actions:
    print("Unexpected shapes found in the following actions:")
    for action in unexpected_actions:
        print(action)
else:
    print("No unexpected shapes found.")

# Convert sequences to NumPy array
X = np.array(sequences)
print("X shape:", X.shape)

y = to_categorical(labels).astype(int)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(50,1662)))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=1000, validation_split=0.2, callbacks=[tb_callback])

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert predicted probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true_labels, y_pred_labels)
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred, average='macro', multi_class='ovo')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)