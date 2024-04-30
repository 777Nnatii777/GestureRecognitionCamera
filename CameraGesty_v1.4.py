
#install python najlepiej mniej niz 12 bo nie dia³a ale wiêcej niz 9 min 10

#instalacja odpowiedniego pakietu pip do wersji pythona

#!pip install opencv-python

#(a jak niedzia³a to innych pip pakietów bo to zalezy od wersji pythona i pip jednam oze dzia³ac 2 nie)

#w koñcu zaimportowanei pakietu co dzia³a ( jak 'import cv2' nie dia³a to ten zaleznie od wrsji pip i lokalizacji mozna uzyc 'from cv2 import cv2')

#!pip install mediapipe

#importowanie mediapipe jako mp dla skrótu
 
#!pip install requests
 
# !wget -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
 
import time

import requests
 
url = f'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task'  # Replace with the actual URL

response = requests.get(url)
 
if response.status_code == 200:

    mab = response.content

else:

    print(f"Failed to fetch the file. Status code: {response.status_code}")
   
 
import sys

sys.getsizeof(mab)
 
flag = 1
 
detectN='Fuck'
 
import mediapipe as mp
 
BaseOptions = mp.tasks.BaseOptions

GestureRecognizer = mp.tasks.vision.GestureRecognizer

GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

VisionRunningMode = mp.tasks.vision.RunningMode
 
# Create a gesture recognizer instance with the live stream mode:

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):

    if result.gestures: 

        global detectN 

        detectN = result.gestures[0][0].category_name 

        print('gesture recognition result: {}'.format(result.gestures))
         
 
options = GestureRecognizerOptions(

    base_options=BaseOptions(model_asset_buffer=mab),

    running_mode=VisionRunningMode.LIVE_STREAM,

    result_callback=print_result)
 
 
import mediapipe as mp
 
BaseOptions = mp.tasks.BaseOptions

GestureRecognizer = mp.tasks.vision.GestureRecognizer

GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

VisionRunningMode = mp.tasks.vision.RunningMode
 
# Create a gesture recognizer instance with the live stream mode:

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):

    if result.gestures: 

        global detectN 

        detectN = result.gestures[0][0].category_name 

        print('gesture recognition result: {}'.format(result.gestures))
 
options = GestureRecognizerOptions(

    base_options=BaseOptions(model_asset_buffer=mab),

    running_mode=VisionRunningMode.LIVE_STREAM,

    result_callback=print_result)
 
import math

from turtle import position

import cv2

import mediapipe as mp

import time
 






#import pyautogui

#protokol do mediapipe na wykrywanie gestow

font                   = cv2.FONT_HERSHEY_SIMPLEX

bottomLeftCornerOfText = (10,40)

fontScale              = 1

fontColor              = (255,0,0)

thickness              = 2

lineType               = 2
 
 


#to 0 oznacza kamerke gdzie 0 to wbudowana a 1,2,3 itd.. to extencion webcams

webcam=cv2.VideoCapture(0)

mp_hands=mp.solutions.hands

mp_drawing=mp.solutions.drawing_utils
 


#definijuemy za³¹czenie siê kamery oraz komendy to zwolnienia kamerki dla innych prgramow bo x nie dzia³a trzeba uzyæ r

while True:
     
    ret, frame=webcam.read()
    time.sleep(0.05)
    

 
    if ret==True:
 
      image_width, image_height, ret = frame.shape
 


        #dodawanie hand traking do funkcji kamery camera czyta BGR a program RGB to tzreba skonwertowaæ a potem w 2 stronê

    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 

    
    #podobno naprawia porblem z ramem idk
    with GestureRecognizer.create_from_options(options) as recognizer:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            recognizer.recognize_async(mp_image, 100)

    if flag==1:  

        results=mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.7,min_tracking_confidence=0.5).process(frame)

      # Convert RGB image to MediaPipe Image format

    with GestureRecognizer.create_from_options(options) as recognizer:

           mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

           recognizer.recognize_async(mp_image, 100)

         # recognition_result = recognizer.recognize(mp_image)

         # if recognition_result.gestures:

            #if recognition_result.gestures[0][0].index>-1:

          #      print(recognition_result.gestures[0][0].index);

        #malowanie lini na palcach spowrotem do BGR zeby program czytal
    

    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    if flag==1:

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(frame,hand_landmarks,connections=mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark

                for id, landmark in enumerate(landmarks):

                    x = int(landmark.x * image_width)

                    y = int(landmark.y *  image_height)
 
                    if id == 8:

                        cv2.circle(img=frame,center=(x + 50,y - 100),radius=8,color=(0,255,255),thickness=2)

                    if id == 4:

                        cv2.circle(img=frame,center=(x + 50,y - 100),radius=8,color=(0,0,255),thickness=2)
                        
                       

    cv2.putText(frame,detectN, bottomLeftCornerOfText, font, fontScale,fontColor,thickness,lineType)  

#tu powyzej jest opcja definiowania jak ma wykrywac czyli jak bardzi musi byc widac ze to reka i jak wyraznie podaac za reka , oraz iel maxrak naraz moze czytac, tu mam lekki problem bo ja duzo rak naraz bedzie to wywala cos cos nie dzia³a

    flag=1
  
    cv2.imshow("CameraNTM",frame)

    key=cv2.waitKey(1) 
    
    
    
    if key==ord("r"):

            break
    


webcam.release()

cv2.destroyAllWindows()
