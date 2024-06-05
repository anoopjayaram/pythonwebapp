from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
from gtts import gTTS
import os
import pygame
import tempfile
import threading

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('gesture_detector_model.h5')

# Initialize Pygame mixer
try:
    pygame.mixer.init()
except pygame.error as e:
    print("Error initializing Pygame mixer:", e)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global variables for gesture tracking
old_gesture = ''
current_gesture = ''
lock = threading.Lock()

def play_audio(text):
    # Initialize gTTS (Google Text-to-Speech) with the text
    tts = gTTS(text, lang='en')

    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        tts.write_to_fp(tmp_file)
        audio_file_path = tmp_file.name

    # Initialize Pygame mixer if not already initialized
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    # Load the audio file with Pygame
    pygame.mixer.music.load(audio_file_path)

    # Play the audio
    pygame.mixer.music.play()

    # Wait until the audio has finished playing
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    # Stop the Pygame mixer to release the file handle
    pygame.mixer.music.stop()

    # Attempt to remove the temporary file
    try:
        os.remove(audio_file_path)
    except Exception as e:
        print("Error occurred while removing temporary file:", e)

def extract_hand_landmarks(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]  # Return the hand landmarks object
    else:
        return None

def process_frame(frame):
    global old_gesture, current_gesture
    hand_landmarks = extract_hand_landmarks(frame)

    # Draw landmarks and display coordinates
    if hand_landmarks:
        # Draw landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Make prediction if hand landmarks are detected
    if hand_landmarks is not None:
        # Convert hand landmarks to numpy array for easier manipulation
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

        # Predict gesture
        prediction = model.predict(np.expand_dims(landmarks_array, axis=0))
        predicted_class = np.argmax(prediction)  # Get the index of the highest probability class

        # Map predicted class index to gesture label
        if predicted_class == 0:
            current_gesture = "Thumbs Up"
        elif predicted_class == 1:
            current_gesture = "Thumbs Down"
        elif predicted_class == 2:
            current_gesture = "Victory"
        elif predicted_class == 3:
            current_gesture = "Non-Thumbs"
        else:
            current_gesture = "Unknown"

        # Display predicted gesture
        cv2.putText(frame, current_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Play audio if gesture has changed
        if current_gesture != old_gesture:
            # Stop the previous audio playback
            pygame.mixer.music.stop()
            # Start a new thread for audio playback
            audio_thread = threading.Thread(target=play_audio, args=(current_gesture,))
            audio_thread.start()
            # Update the old gesture
            old_gesture = current_gesture

    return frame

def generate_frames():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Process the frame
            with lock:
                frame = process_frame(frame)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

