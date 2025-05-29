import streamlit as st
import mediapipe as mp
import cv2
import time
from PIL import Image
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision




st.sidebar.title('Sign Language Detection ')


app_mode = st.sidebar.selectbox('Choose the App mode',
['Sign Language to Text','Text to sign Language'])



if app_mode == 'Sign Language to Text':
    st.title('Sign Language to Text')

    st.sidebar.markdown('---')
    
    st.markdown(' ## Output')
    

    stframe = st.empty()


    cam=cv2.VideoCapture(0)

    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))  

    st.markdown("<hr/>", unsafe_allow_html=True)

    
     
    base_options = python.BaseOptions(model_asset_buffer = open("gesture_recognize1r.task", "rb").read())
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    cam=cv2.VideoCapture(0)
    while True:
        success, frame = cam.read()
        if not success:
            print("Failed to capture image")
            continue
        
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        recognition_result = recognizer.recognize(image)

        if not recognition_result.gestures:
            gesture="None"
        else:
            gesture = recognition_result.gestures[0][0].category_name
            
        
        frame=cv2.putText(frame, gesture, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        

        
        stframe.image(frame, channels='BGR', use_container_width=True)

else:
    st.title('Text to Sign Language')


    
    def display_images(text):
        
        img_dir = "images/"

        
        image_pos = st.empty()

        
        for char in text:
            if char.isalpha():
                
                img_path = os.path.join(img_dir, f"{char}.png")
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

    
    display_images(text)
