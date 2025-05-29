import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from PIL import Image
import os

st.sidebar.title('Sign Language Detection')

app_mode = st.sidebar.selectbox('Choose the App mode',
                                 ['Sign Language to Text (Live)', 'Text to sign Language'])

if app_mode == 'Sign Language to Text (Live)':
    st.title('Sign Language to Text - Live Mode')

    # Load model once globally
    base_options = python.BaseOptions(model_asset_buffer=open("gesture_recognize1r.task", "rb").read())
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            recognition_result = recognizer.recognize(mp_image)
            gesture = "None"
            if recognition_result.gestures:
                gesture = recognition_result.gestures[0][0].category_name

            image = cv2.putText(image, gesture, (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            return image

    webrtc_streamer(key="sign-lang", video_transformer_factory=VideoTransformer)

else:
    st.title('Text to Sign Language')

    def display_images(text):
        img_dir = "images/"
        image_pos = st.empty()

        for char in text:
            if char.isalpha():
                img_path = os.path.join(img_dir, f"{char}.png")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    image_pos.image(img, width=500)
                    time.sleep(2)
                    image_pos.empty()
            else:
                time.sleep(1)
                image_pos.empty()

        time.sleep(2)
        image_pos.empty()

    text = st.text_input("Enter text:")
    text = text.lower()

    if text:
        display_images(text)
