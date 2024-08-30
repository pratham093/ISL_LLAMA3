import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from openai import OpenAI
import tkinter as tk
from tkinter import scrolledtext

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model = tf.keras.models.load_model('best_model.keras')

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def sug_corr(input_words):
    message = f"Create a sentence using the following words: {', '.join(input_words)}. Ensure that the sentence is meaningful even if some words seem incorrect or out of context. Create sentences out of the list of words just make something closest possible meaning. only make one sentence. Just one sentence dont talk anything else I just want the sentence. One Sentence, dont't come up with more possiblities."
    
    print("Sending request to Llama3 model...")
    try:
        completion = client.completions.create(
            model="local-model",  # Ensure the correct model name
            prompt=message,
            temperature=0.7,
            max_tokens=100,  # Increase the output length to avoid truncation issues
            stream=False    # Disable streaming for simplicity
        )

        # Debugging: Print the full completion response
        print(f"Completion response: {completion}")

        suggestions = []
        if completion and 'choices' in completion and completion['choices']:
            # Properly extract the text content
            text = completion['choices'][0]['text'].strip()
            if text:
                suggestions.append(text)
                print(f"Received text: {text}")
            else:
                print("Received an empty text in the completion response.")
        else:
            print("No valid choices found in the completion response.")

    except Exception as e:
        print(f"Error generating sentence: {e}")
        return None

    return suggestions[0] if suggestions else "Unable to generate a sentence with the given actions."

# Initialize the OpenAI client
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Path for actions
actions = np.array([
    'Beach', 'Blue', 'Car', 'Dance', 'Deaf',  'Family', 'Flower',
    'Friend', 'Happy', 'Hello', 'Help', 'I', 'Jump', 'Laugh',
    'Man', 'Play', 'Please', 'Red', 'Restaurant', 'Run', 'Sit',
    'Sorry', 'Stand', 'Stop', 'Thanks',  'Wait',
    'Woman', 'You'
])
# Initialize the MediaPipe holistic model
holistic = mp_holistic.Holistic()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set frame rate and sequence length
fps = 30
sequence_length = 50

# Recording control variables
recording = False
frames = []
sequence = []

# List to store detected actions
detected_actions = []

# Variable to track the last detected action
last_action = None

# Variable to store the generated sentence
generated_sentence = ""

# Initialize tkinter GUI
root = tk.Tk()
root.title("Sign Language Detection and Sentence Generation")

# Create text display area for detected actions
text_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10, font=("Arial", 12))
text_display.grid(column=0, row=0, padx=10, pady=10)

def display_text(text):
    """Displays text in the scrolled text widget."""
    text_display.insert(tk.END, text + "\n")
    text_display.see(tk.END)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame. Exiting...")
        break

    # Convert the image color for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    if recording:
        # Draw landmarks on the image
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) == sequence_length:
            # Convert sequence to numpy array
            X = np.array(sequence)
            X = np.expand_dims(X, axis=0)  # Add batch dimension

            # Make prediction
            y_pred = model.predict(X)
            
            # Get the predicted action and its confidence score
            action_index = np.argmax(y_pred, axis=1)[0]
            confidence = y_pred[0][action_index]

            # Check if confidence is above the threshold
            if confidence >= 0.9:
                action = actions[action_index]
                
                # Check if the detected action is different from the last detected action
                if action != last_action:
                    detected_actions.append(action)
                    last_action = action
                    print(f"Predicted action: {action} with confidence {confidence:.2f}")
                    display_text(f"Predicted action: {action} (Confidence: {confidence:.2f})")

            # Remove the oldest frame from the sequence
            sequence.pop(0)

        # Store the frame in the frames list
        frames.append(frame)

    # Display the frame
    cv2.imshow('Sign Language Detection', frame)

    # Check for user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        # Start recording
        recording = True
        print("Recording started...")
        display_text("Recording started...")
        frames = []  # Reset frames list
    elif key == ord('s') and recording:
        # Stop recording
        recording = False
        print("Recording stopped.")
        display_text("Recording stopped.")
        break
    elif key == ord('q'):
        # Quit the program
        print("Quitting the program...")
        display_text("Quitting the program...")
        break

# Release resources after recording or quitting
cap.release()
cv2.destroyAllWindows()

# Generate the sentence from detected actions using Llama3
if detected_actions:
    print(f"Detected actions: {detected_actions}")
    display_text(f"Detected actions: {', '.join(detected_actions)}")
    generated_sentence = sug_corr(detected_actions)
    if generated_sentence:
        print(f"Generated sentence: {generated_sentence}")
        display_text(f"Generated sentence: {generated_sentence}")
    else:
        print("Failed to generate a valid sentence.")
        display_text("Failed to generate a valid sentence.")
else:
    print("No actions were detected.")
    display_text("No actions were detected.")

# Start the tkinter main loop
root.mainloop()
