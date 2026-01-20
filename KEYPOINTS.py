import cv2
import numpy as np
import os
import mediapipe as mp
import argparse

try:
    from config import ACTIONS, SEQUENCE_LENGTH, DATA_PATH, NUM_SEQUENCES
except ImportError:
    DATA_PATH = 'Final_Data'
    SEQUENCE_LENGTH = 50
    NUM_SEQUENCES = 70
    ACTIONS = [
        'Beach', 'Blue', 'Car', 'Dance', 'Deaf', 'Family', 'Flower',
        'Friend', 'Happy', 'Hello', 'Help', 'I', 'Jump', 'Laugh',
        'Man', 'Play', 'Please', 'Red', 'Restaurant', 'Run', 'Sit',
        'Sorry', 'Stand', 'Stop', 'Thanks', 'Wait', 'Woman', 'Work', 'You'
    ]

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def get_existing_sequences(action_path):
    if not os.path.exists(action_path):
        return 0
    return len([d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))])


def collect_single_action(action, num_sequences, data_path, sequence_length):
    os.makedirs(data_path, exist_ok=True)
    action_path = os.path.join(data_path, action)
    os.makedirs(action_path, exist_ok=True)
    
    existing = get_existing_sequences(action_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    paused = False
    first_sequence = True
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for sequence in range(existing, existing + num_sequences):
            sequence_path = os.path.join(action_path, str(sequence))
            os.makedirs(sequence_path, exist_ok=True)
            
            if first_sequence:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
                    cv2.putText(frame, f'GET READY: {action}', (w//2 - 150, h//2 - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, f'{num_sequences} sequences', (w//2 - 100, h//2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, 'Press any key to start...', (w//2 - 130, h//2 + 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(0)
                first_sequence = False
            
            frame_num = 0
            while frame_num < sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                
                draw_landmarks(frame, results)
                
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
                
                if paused:
                    cv2.rectangle(frame, (0, h//2 - 40), (w, h//2 + 40), (0, 0, 0), -1)
                    cv2.putText(frame, 'PAUSED', (w//2 - 80, h//2 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.putText(frame, 'P = resume | R = restart sequence | Q = quit', (w//2 - 220, h//2 + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow('Data Collection', frame)
                    
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('p'):
                        paused = False
                    elif key == ord('r'):
                        paused = False
                        frame_num = 0
                        for f in os.listdir(sequence_path):
                            os.remove(os.path.join(sequence_path, f))
                    elif key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return False
                    continue
                
                progress_total = ((sequence - existing) * sequence_length + frame_num) / (num_sequences * sequence_length)
                bar_width = int(progress_total * (w - 20))
                cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, 50), (w - 10, 70), (255, 255, 255), 2)
                
                cv2.putText(frame, f'{action} | Seq: {sequence - existing + 1}/{num_sequences} | Frame: {frame_num + 1}/{sequence_length}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.rectangle(frame, (0, h-30), (w, h), (0, 0, 0), -1)
                cv2.putText(frame, 'P = pause | Q = quit', (w//2 - 80, h-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
                cv2.imshow('Data Collection', frame)
                
                keypoints = extract_keypoints(results)
                np.save(os.path.join(sequence_path, f'{frame_num}.npy'), keypoints)
                
                frame_num += 1
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
                elif key == ord('p'):
                    paused = True
            
            print(f"  Sequence {sequence - existing + 1}/{num_sequences} done")
    
    cap.release()
    cv2.destroyAllWindows()
    return True


def collect_multiple_actions(actions, num_sequences, data_path, sequence_length):
    print(f"\nCollecting {len(actions)} actions")
    print(f"Sequences per action: {num_sequences}")
    print(f"Frames per sequence: {sequence_length}")
    
    for i, action in enumerate(actions):
        action_path = os.path.join(data_path, action)
        existing = get_existing_sequences(action_path)
        
        if existing >= num_sequences:
            print(f"\n[{i+1}/{len(actions)}] {action}: DONE ({existing} sequences)")
            continue
        
        remaining = num_sequences - existing
        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(actions)}] {action}")
        print(f"Existing: {existing} | Need: {remaining} more")
        print(f"{'='*50}")
        
        success = collect_single_action(action, remaining, data_path, sequence_length)
        if not success:
            print(f"\nStopped at action: {action}")
            break
    
    print(f"\nData collection complete. Saved to {data_path}")


def interactive_mode():
    print("\n" + "="*50)
    print("ISL Data Collection - Interactive Mode")
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("  1. Add new action")
        print("  2. Add more sequences to existing action")
        print("  3. List existing actions")
        print("  4. Quit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == '1':
            action = input("Enter action name: ").strip()
            if not action:
                continue
            num_seq = input(f"Number of sequences [{NUM_SEQUENCES}]: ").strip()
            num_seq = int(num_seq) if num_seq else NUM_SEQUENCES
            collect_single_action(action, num_seq, DATA_PATH, SEQUENCE_LENGTH)
            
        elif choice == '2':
            if not os.path.exists(DATA_PATH):
                print("No data folder found")
                continue
            actions = sorted([d for d in os.listdir(DATA_PATH) 
                            if os.path.isdir(os.path.join(DATA_PATH, d))])
            if not actions:
                print("No existing actions")
                continue
            print("\nExisting actions:")
            for i, a in enumerate(actions):
                count = get_existing_sequences(os.path.join(DATA_PATH, a))
                print(f"  {i+1}. {a} ({count} sequences)")
            idx = input("Select action number: ").strip()
            try:
                action = actions[int(idx) - 1]
                num_seq = input(f"Additional sequences [{NUM_SEQUENCES}]: ").strip()
                num_seq = int(num_seq) if num_seq else NUM_SEQUENCES
                collect_single_action(action, num_seq, DATA_PATH, SEQUENCE_LENGTH)
            except (ValueError, IndexError):
                print("Invalid selection")
                
        elif choice == '3':
            if not os.path.exists(DATA_PATH):
                print("No data folder found")
                continue
            actions = sorted([d for d in os.listdir(DATA_PATH) 
                            if os.path.isdir(os.path.join(DATA_PATH, d))])
            print(f"\nActions in {DATA_PATH}:")
            total = 0
            for a in actions:
                count = get_existing_sequences(os.path.join(DATA_PATH, a))
                total += count
                print(f"  {a}: {count} sequences")
            print(f"\nTotal: {len(actions)} actions, {total} sequences")
            
        elif choice == '4':
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect ISL training data')
    parser.add_argument('--actions', nargs='+', help='List of actions to collect')
    parser.add_argument('--sequences', type=int, default=NUM_SEQUENCES, help='Sequences per action')
    parser.add_argument('--frames', type=int, default=SEQUENCE_LENGTH, help='Frames per sequence')
    parser.add_argument('--data-path', type=str, default=DATA_PATH, help='Output data path')
    parser.add_argument('--menu', '-m', action='store_true', help='Menu mode')
    
    args = parser.parse_args()
    
    if args.menu:
        interactive_mode()
    elif args.actions:
        collect_multiple_actions(args.actions, args.sequences, args.data_path, args.frames)
    else:
        print("\n" + "="*50)
        print("ISL Data Collection")
        print("="*50)
        print(f"\nActions from config: {len(ACTIONS)}")
        print(f"Sequences per action: {NUM_SEQUENCES}")
        print(f"Frames per sequence: {SEQUENCE_LENGTH}")
        print(f"Data path: {DATA_PATH}")
        print("\nControls: P=pause, R=redo, Q=quit")
        print("="*50)
        input("\nPress ENTER to start...")
        collect_multiple_actions(ACTIONS, NUM_SEQUENCES, DATA_PATH, SEQUENCE_LENGTH)
