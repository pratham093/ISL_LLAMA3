import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional,
    Input, Masking, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = 'Final_Data'
MODEL_NAME = 'final_model.keras'
SEQUENCE_LENGTH = 50
KEYPOINT_SIZE = 1662


def get_actions_from_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found")
        return []
    actions = sorted([d for d in os.listdir(DATA_PATH) 
                     if os.path.isdir(os.path.join(DATA_PATH, d))])
    return actions


def normalize_keypoints(keypoints):
    """
    Normalize a single frame's keypoints.
    - Centers on mid-shoulder
    - Scales by shoulder width
    """
    pose_end = 33 * 4
    face_end = pose_end + 468 * 3
    lh_end = face_end + 21 * 3
    
    pose = keypoints[:pose_end].reshape(33, 4)
    face = keypoints[pose_end:face_end].reshape(468, 3)
    lh = keypoints[face_end:lh_end].reshape(21, 3)
    rh = keypoints[lh_end:].reshape(21, 3)
    
    left_shoulder = pose[11, :3]
    right_shoulder = pose[12, :3]
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    
    shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
    scale = shoulder_width if shoulder_width > 0.01 else 1.0
    
    pose_norm = (pose[:, :3] - mid_shoulder) / scale
    face_norm = (face - mid_shoulder) / scale
    lh_norm = (lh - mid_shoulder) / scale
    rh_norm = (rh - mid_shoulder) / scale
    
    lh_present = 1.0 if np.any(lh != 0) else 0.0
    rh_present = 1.0 if np.any(rh != 0) else 0.0
    
    return np.concatenate([
        pose_norm.flatten(),
        face_norm.flatten(),
        lh_norm.flatten(),
        rh_norm.flatten(),
        [lh_present, rh_present]
    ])


def add_velocity(sequence):
    """Add velocity (frame differences) as features."""
    velocity = np.diff(sequence, axis=0, prepend=sequence[[0]])
    return np.concatenate([sequence, velocity], axis=1)


def smooth_sequence(sequence, window=3):
    """Moving average smoothing."""
    smoothed = np.zeros_like(sequence)
    for i in range(len(sequence)):
        start = max(0, i - window // 2)
        end = min(len(sequence), i + window // 2 + 1)
        smoothed[i] = np.mean(sequence[start:end], axis=0)
    return smoothed


def preprocess_sequence(sequence, normalize=True, velocity=False, smooth=True):
    """Full preprocessing pipeline."""
    seq = np.array(sequence)
    
    if normalize:
        seq = np.array([normalize_keypoints(frame) for frame in seq])
    
    if smooth:
        seq = smooth_sequence(seq, window=3)
    
    if velocity:
        seq = add_velocity(seq)
    
    return seq


def augment_sequence(sequence):
    """Data augmentation with multiple techniques."""
    seq = sequence.copy()
    
    # Always apply small noise
    if np.random.random() < 0.5:
        noise = np.random.normal(0, 0.01, seq.shape)
        seq = seq + noise
    
    # Random scaling
    if np.random.random() < 0.3:
        scale = np.random.uniform(0.95, 1.05)
        seq = seq * scale
    
    # Random time shift (drop first/last few frames and pad)
    if np.random.random() < 0.2:
        shift = np.random.randint(1, 4)
        if np.random.random() < 0.5:
            seq = np.concatenate([seq[shift:], np.tile(seq[[-1]], (shift, 1))], axis=0)
        else:
            seq = np.concatenate([np.tile(seq[[0]], (shift, 1)), seq[:-shift]], axis=0)
    
    return seq


def load_data(preprocess=True, augment=True, aug_factor=2):
    actions = get_actions_from_data()
    if not actions:
        return None, None, None, None
    
    print(f"Found {len(actions)} actions: {actions}")
    
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        seq_dirs = sorted([d for d in os.listdir(action_path) 
                          if os.path.isdir(os.path.join(action_path, d))])
        
        action_sequences = []
        for sequence_dir in seq_dirs:
            sequence_path = os.path.join(action_path, sequence_dir)
            window = []
            
            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(sequence_path, f'{frame_num}.npy')
                if os.path.exists(frame_path):
                    keypoints = np.load(frame_path)
                    if keypoints.shape[0] == KEYPOINT_SIZE:
                        window.append(keypoints)
            
            if len(window) == SEQUENCE_LENGTH:
                action_sequences.append(np.array(window))
        
        print(f"  {action}: {len(action_sequences)} sequences")
        
        for seq in action_sequences:
            if preprocess:
                seq = preprocess_sequence(seq)
            sequences.append(seq)
            labels.append(label_map[action])
            
            if augment:
                for _ in range(aug_factor):
                    aug_seq = augment_sequence(seq.copy())
                    sequences.append(aug_seq)
                    labels.append(label_map[action])
    
    with open('label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    X = np.array(sequences)
    y = np.array(labels)
    feature_size = X.shape[2]
    
    return X, y, label_map, feature_size


def build_model(num_classes, input_shape):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))),
        BatchNormalization(),
        Dropout(0.5),
        
        Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01))),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    return model


def plot_training_history(history, filename='training_history.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['categorical_accuracy'], label='Train')
    axes[0].plot(history.history['val_categorical_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(12, len(classes)), max(10, len(classes) * 0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def train(preprocess=True, augment=True, aug_factor=2, epochs=300, batch_size=32):
    print("="*60)
    print("ISL Model Training")
    print("="*60)
    print(f"\nPreprocessing: {preprocess}")
    print(f"  - Normalization (center + scale)")
    print(f"  - Smoothing (reduce jitter)")
    print(f"  - Velocity features (motion)")
    print(f"Augmentation: {augment} (x{aug_factor + 1})")
    print(f"Epochs: {epochs}")
    print("="*60)
    
    print("\nLoading data...")
    X, y, label_map, feature_size = load_data(
        preprocess=preprocess, 
        augment=augment, 
        aug_factor=aug_factor
    )
    
    if X is None:
        print("No data loaded")
        return None, None
    
    print(f"\nTotal samples: {X.shape[0]}")
    print(f"Sequence shape: {X.shape[1:]}")
    print(f"Feature size: {feature_size}")
    print(f"Classes: {len(label_map)}")
    
    y_cat = to_categorical(y, num_classes=len(label_map))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    model = build_model(len(label_map), (X.shape[1], X.shape[2]))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['categorical_accuracy']
    )
    model.summary()
    
    callbacks = [
        TensorBoard(log_dir='logs'),
        EarlyStopping(monitor='val_loss', patience=30, 
                     restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_NAME, monitor='val_loss', 
                       save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, 
                         min_lr=1e-6, verbose=1)
    ]
    
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    
    plot_training_history(history)
    
    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
    
    print(f"\n{'='*40}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'='*40}")
    
    classes = list(label_map.keys())
    print("\nPer-class Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=classes, zero_division=0))
    
    plot_confusion_matrix(y_true_labels, y_pred_labels, classes)
    
    model.save(MODEL_NAME)
    print(f"\nSaved: {MODEL_NAME}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-preprocess', action='store_true')
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--aug-factor', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    train(
        preprocess=not args.no_preprocess,
        augment=not args.no_augment,
        aug_factor=args.aug_factor,
        epochs=args.epochs,
        batch_size=args.batch_size
    )