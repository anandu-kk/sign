import streamlit as st
import mediapipe as mp
import cv2
import time
from PIL import Image
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoFrame
import numpy as np

# Load the gesture recognizer model once
# It's good practice to load models outside the main loop
# to avoid reloading on every rerun
@st.cache_resource
def load_gesture_recognizer_model():
    # Make sure 'gesture_recognize1r.task' is in the same directory or accessible path
    base_options = python.BaseOptions(model_asset_buffer=open("gesture_recognize1r.task", "rb").read())
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    return recognizer

recognizer = load_gesture_recognizer_model()

st.sidebar.title('Sign Language Detection')

app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['Sign Language to Text', 'Text to Sign Language'])

if app_mode == 'Sign Language to Text':
    st.title('Sign Language to Text')

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    # Define the VideoProcessor for streamlit-webrtc
    class MediaPipeGestureRecognizer(VideoProcessorBase):
        def recv(self, frame: VideoFrame) -> VideoFrame:
            # Convert WebRTC frame to OpenCV format
            img = frame.to_ndarray(format="bgr24")

            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image object
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Perform gesture recognition
            recognition_result = recognizer.recognize(image)

            gesture = "None"
            if recognition_result.gestures:
                # Get the top gesture
                gesture = recognition_result.gestures[0][0].category_name

            # Draw the gesture on the frame
            cv2.putText(img, gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

            # Draw hand landmarks if available (optional, but good for visualization)
            if recognition_result.hand_landmarks:
                for hand_landmarks in recognition_result.hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        img,
                        mp.solutions.hands.HandLandmarks(hand_landmarks),
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style())

            # Return the processed frame
            return VideoFrame.from_ndarray(img, format="bgr24")

    # Use webrtc_streamer to get video input from the user's browser
    webrtc_streamer(
        key="sign-language-detector",
        video_processor_factory=MediaPipeGestureRecognizer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Recommended for potentially slow processing
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

else:
    st.title('Text to Sign Language')

    def display_images(text):
        img_dir = "images/" # Ensure this directory exists relative to your app

        # Use a placeholder for dynamic image updates
        image_placeholder = st.empty()

        for char in text:
            if char.isalpha():
                img_path = os.path.join(img_dir, f"{char}.png")
                if os.path.exists(img_path): # Check if image exists
                    img = Image.open(img_path)
                    image_placeholder.image(img, width=500)
                    time.sleep(2) # Display for 2 seconds
                else:
                    image_placeholder.warning(f"Image for '{char}' not found.")
                    time.sleep(1)
            else:
                # Clear the image for non-alphabetic characters or pauses
                image_placeholder.empty()
                time.sleep(1)

        # Ensure the image placeholder is empty after the loop
        image_placeholder.empty()

    text_input = st.text_input("Enter text:")
    # Ensure text is lowercase for consistent image lookup
    processed_text = text_input.lower()

    if st.button("Display Signs"):
        display_images(processed_text)
