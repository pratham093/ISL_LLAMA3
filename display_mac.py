#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from collections import deque
from typing import Optional, List

try:
    from config import MODEL_PATH, ACTIONS, SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MAX_ACTIONS, COOLDOWN_SECONDS, STABILITY_COUNT
except ImportError:
    MODEL_PATH = 'final_model.keras'
    ACTIONS = [
        'Beach', 'Blue', 'Car', 'Dance', 'Deaf', 'Family', 'Flower',
        'Friend', 'Happy', 'Hello', 'Help', 'I', 'Jump', 'Laugh',
        'Man', 'Play', 'Please', 'Red', 'Restaurant', 'Run', 'Sit',
        'Sorry', 'Stand', 'Stop', 'Thanks', 'Wait', 'Woman', 'Work', 'You'
    ]
    SEQUENCE_LENGTH = 20
    CONFIDENCE_THRESHOLD = 0.9
    LLM_BASE_URL = "http://localhost:8000/v1"
    LLM_API_KEY = "not-needed"
    LLM_MODEL = "local-model"
    MAX_ACTIONS = 10
    COOLDOWN_SECONDS = 1.5
    STABILITY_COUNT = 2


def setup_mediapipe():
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'holistic'):
            print("Using MediaPipe solutions API")
            return mp, mp.solutions.holistic, mp.solutions.drawing_utils
        raise AttributeError("solutions not available")
    except (AttributeError, ImportError) as e:
        try:
            from mediapipe.python.solutions import holistic as mp_holistic
            from mediapipe.python.solutions import drawing_utils as mp_drawing
            import mediapipe as mp
            print("Using MediaPipe python.solutions API")
            return mp, mp_holistic, mp_drawing
        except ImportError:
            pass
        print("MediaPipe installation issue. Run: pip install mediapipe==0.10.9")
        sys.exit(1)


mp, mp_holistic, mp_drawing = setup_mediapipe()


class Config:
    MODEL_PATH = MODEL_PATH
    LLM_BASE_URL = LLM_BASE_URL
    LLM_API_KEY = LLM_API_KEY
    LLM_MODEL = LLM_MODEL
    SEQUENCE_LENGTH = SEQUENCE_LENGTH
    CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    ACTIONS = ACTIONS
    MAX_ACTIONS = MAX_ACTIONS
    COOLDOWN_SECONDS = COOLDOWN_SECONDS
    STABILITY_COUNT = STABILITY_COUNT


class KeypointExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.POSE_CONNECTIONS = getattr(mp_holistic, 'POSE_CONNECTIONS', None)
        self.FACEMESH_CONTOURS = getattr(mp_holistic, 'FACEMESH_CONTOURS', 
                                          getattr(mp_holistic, 'FACE_CONNECTIONS', None))
        self.HAND_CONNECTIONS = getattr(mp_holistic, 'HAND_CONNECTIONS', None)
    
    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        return results
    
    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten() \
               if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] 
                        for res in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten() \
             if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten() \
             if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])
    
    def normalize_keypoints(self, keypoints):
        pose_end = 33 * 4
        face_end = pose_end + 468 * 3
        lh_end = face_end + 21 * 3
        
        pose = keypoints[:pose_end].reshape(33, 4)
        lh = keypoints[face_end:lh_end].reshape(21, 3)
        rh = keypoints[lh_end:].reshape(21, 3)
        
        left_shoulder = pose[11, :3]
        right_shoulder = pose[12, :3]
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        
        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        scale = shoulder_width if shoulder_width > 0.01 else 1.0
        
        pose_norm = (pose[:, :3] - mid_shoulder) / scale
        lh_norm = (lh - mid_shoulder) / scale
        rh_norm = (rh - mid_shoulder) / scale
        
        lh_present = 1.0 if np.any(lh != 0) else 0.0
        rh_present = 1.0 if np.any(rh != 0) else 0.0
        
        return np.concatenate([
            pose_norm.flatten(),
            lh_norm.flatten(),
            rh_norm.flatten(),
            [lh_present, rh_present]
        ])
    
    def preprocess_sequence(self, sequence):
        seq = np.array([self.normalize_keypoints(frame) for frame in sequence])
        smoothed = np.zeros_like(seq)
        for i in range(len(seq)):
            start = max(0, i - 1)
            end = min(len(seq), i + 2)
            smoothed[i] = np.mean(seq[start:end], axis=0)
        return smoothed
    
    def draw_landmarks(self, frame, results):
        if self.POSE_CONNECTIONS:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.POSE_CONNECTIONS)
        if self.HAND_CONNECTIONS:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.HAND_CONNECTIONS)
        return frame
    
    def close(self):
        self.holistic.close()


class SentenceGenerator:
    def __init__(self, base_url, api_key, model):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model = model
            self.available = True
        except ImportError:
            print("OpenAI library not installed. Using simple concatenation.")
            self.available = False
    
    def generate_sentence(self, words):
        if not words:
            return None
        
        if not self.available:
            return " ".join(words).capitalize() + "."
        
        prompt = f"Create one short meaningful sentence using these words: {', '.join(words)}. Output only the sentence."
        
        print(f"\nGenerating sentence from: {words}")
        
        try:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Create one short sentence from given words. Output only the sentence."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=50
                )
                sentence = response.choices[0].message.content.strip()
            except Exception:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=50
                )
                sentence = response.choices[0].text.strip()
            
            print(f"Generated: {sentence}")
            return sentence
        except Exception as e:
            print(f"LLM error: {e}")
            return " ".join(words).capitalize() + "."


class GestureRecognizer:
    def __init__(self, model_path, actions, sequence_length=50, confidence_threshold=0.7, stability_count=2, keypoint_extractor=None):
        self.model = self._load_model(model_path)
        self.actions = np.array(actions)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.stability_count = stability_count
        self.sequence = deque(maxlen=sequence_length)
        self.last_action = None
        self.last_detection_time = 0
        self.pending_action = None
        self.pending_count = 0
        self.keypoint_extractor = keypoint_extractor
    
    def _load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def add_keypoints(self, keypoints, cooldown_seconds=1.5):
        self.sequence.append(keypoints)
        
        if len(self.sequence) < self.sequence_length:
            return None
        
        current_time = time.time()
        if current_time - self.last_detection_time < cooldown_seconds:
            return None
        
        raw_sequence = list(self.sequence)
        if self.keypoint_extractor:
            X = self.keypoint_extractor.preprocess_sequence(raw_sequence)
        else:
            X = np.array(raw_sequence)
        X = np.expand_dims(X, axis=0)
        
        y_pred = self.model.predict(X, verbose=0)
        action_index = np.argmax(y_pred, axis=1)[0]
        confidence = y_pred[0][action_index]
        
        if confidence >= self.confidence_threshold:
            action = self.actions[action_index]
            
            if action == self.pending_action:
                self.pending_count += 1
            else:
                self.pending_action = action
                self.pending_count = 1
            
            if self.pending_count >= self.stability_count and action != self.last_action:
                self.last_action = action
                self.last_detection_time = current_time
                self.pending_action = None
                self.pending_count = 0
                return (action, confidence)
        
        return None
    
    def reset(self):
        self.sequence.clear()
        self.last_action = None
        self.last_detection_time = 0
        self.pending_action = None
        self.pending_count = 0


class SignLanguageDetector:
    def __init__(self, config):
        self.config = config
        self.detected_actions = []
        self.recording = False
        
        print("\nInitializing...")
        self.keypoint_extractor = KeypointExtractor()
        self.gesture_recognizer = GestureRecognizer(
            model_path=config.MODEL_PATH,
            actions=config.ACTIONS,
            sequence_length=config.SEQUENCE_LENGTH,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            stability_count=config.STABILITY_COUNT,
            keypoint_extractor=self.keypoint_extractor
        )
        self.sentence_generator = SentenceGenerator(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL
        )
    
    def _init_camera(self):
        print("Initializing camera...")
        cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        print("Camera ready!")
        return cap
    
    def _draw_ui(self, frame):
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        
        if self.recording:
            remaining = self.config.MAX_ACTIONS - len(self.detected_actions)
            if remaining > 0:
                status = f"RECORDING ({remaining} left)"
                color = (0, 0, 255)
            else:
                status = "MAX REACHED - Press S"
                color = (0, 165, 255)
            cv2.circle(frame, (25, 25), 10, color, -1)
        else:
            status = "READY"
            color = (128, 128, 128)
            cv2.circle(frame, (25, 25), 10, (128, 128, 128), -1)
        
        cv2.putText(frame, status, (45, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if self.detected_actions:
            words = " > ".join(self.detected_actions)
            cv2.putText(frame, f"Words: {words}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "[R] Record  [S] Stop & Generate  [C] Clear  [Q] Quit", 
                   (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def _generate_and_show(self):
        self.recording = False
        print("\nRecording stopped.")
        
        if self.detected_actions:
            print(f"\nDetected: {self.detected_actions}")
            sentence = self.sentence_generator.generate_sentence(self.detected_actions)
            if sentence:
                print(f"\n>>> {sentence}\n")
        else:
            print("No actions detected.")
    
    def run(self):
        cap = self._init_camera()
        
        print("\n" + "="*50)
        print("ISL Sign Language Detector")
        print("="*50)
        print(f"\nMax words per sentence: {self.config.MAX_ACTIONS}")
        print(f"Cooldown between detections: {self.config.COOLDOWN_SECONDS}s")
        print(f"Confidence threshold: {self.config.CONFIDENCE_THRESHOLD}")
        print("\nControls:")
        print("  [R] Start recording")
        print("  [S] Stop & generate sentence")
        print("  [C] Clear words")
        print("  [Q] Quit")
        print("="*50 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                results = self.keypoint_extractor.process_frame(frame)
                
                if self.recording and len(self.detected_actions) < self.config.MAX_ACTIONS:
                    frame = self.keypoint_extractor.draw_landmarks(frame, results)
                    keypoints = self.keypoint_extractor.extract_keypoints(results)
                    prediction = self.gesture_recognizer.add_keypoints(
                        keypoints, 
                        self.config.COOLDOWN_SECONDS
                    )
                    
                    if prediction:
                        action, confidence = prediction
                        self.detected_actions.append(action)
                        print(f"[{len(self.detected_actions)}/{self.config.MAX_ACTIONS}] {action} ({confidence:.0%})")
                        
                        if len(self.detected_actions) >= self.config.MAX_ACTIONS:
                            print("\nMax words reached! Press [S] to generate sentence.")
                
                frame = self._draw_ui(frame)
                cv2.imshow('ISL Detector', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r'):
                    self.recording = True
                    self.detected_actions = []
                    self.gesture_recognizer.reset()
                    print("\nRecording started...")
                
                elif key == ord('s'):
                    self._generate_and_show()
                
                elif key == ord('c'):
                    self.detected_actions = []
                    self.gesture_recognizer.reset()
                    print("Cleared.")
                
                elif key == ord('q'):
                    print("\nGoodbye!")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.keypoint_extractor.close()


def main():
    print("\n" + "="*50)
    print("Starting ISL to Text System...")
    print("="*50)
    print(f"\nTensorFlow: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {gpus}")
    
    config = Config()
    app = SignLanguageDetector(config)
    app.run()


if __name__ == "__main__":
    main()