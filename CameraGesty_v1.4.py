import time
import requests
import sys
import cv2
import mediapipe as mp
import webbrowser

# Definiujemy stale do wyswietlenia tekstu na ekranie
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 40)
fontScale = 1
fontColor = (255, 0, 0)
thickness = 2
lineType = 2

# Pobieranie modelu
url = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task'
response = requests.get(url)
if response.status_code == 200:
    model_asset_buffer = response.content
else:
    print(f"Failed to fetch the file. Status code: {response.status_code}")
    sys.exit(1)

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_image = mp.Image
mp_image_format = mp.ImageFormat

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Zabezpieczenie przed wielokrotnym wykonywaniem tej samej akcji w krótkim czasie
last_action_time = 0
last_action_name = ''

# Funkcja obslugujaca rózne gesty
def handle_gesture(gesture_name, num_fingers):
    global last_action_time, last_action_name
    current_time = time.time()

    # Sprawdzamy, czy uplynelo co najmniej 10 sekund od ostatniej akcji
    if current_time - last_action_time < 10:
        print("Zabezpieczenie: Poczekaj 10 sekund przed wykonaniem kolejnej akcji.")
        return

    # Sprawdzamy, czy ta sama akcja nie zosta³a wykonana dwukrotnie pod rzad
    if gesture_name == last_action_name:
        print("Zabezpieczenie: Ta sama akcja nie moze byc wykonana dwukrotnie pod rzad.")
        return

    # Wykonujemy akcje
    if gesture_name == 'thumb_up':
        webbrowser.open('https://www.youtube.com')
    elif gesture_name == 'peace':
        webbrowser.open('https://www.google.com')
    elif gesture_name == 'fist':
        print("Detected fist gesture!")
    elif gesture_name == 'okay':
        print("Detected okay gesture!")
    elif gesture_name == 'palm':
        print("Detected palm gesture!")
    elif gesture_name == 'one':
        print("Detected one gesture!")
    elif num_fingers == 2:
        webbrowser.open('https://www.facebook.com')
    elif num_fingers == 3:
        webbrowser.open('https://www.twitter.com')
    elif num_fingers == 4:
        webbrowser.open('https://www.linkedin.com')
    
    else:
        print(f"Unknown gesture: {gesture_name} with {num_fingers} fingers")

    # Aktualizujemy czas ostatniej akcji
    last_action_time = current_time
    last_action_name = gesture_name

# Funkcja do okreœlania liczby wyprostowanych palców
def count_fingers(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]  # Indeksy punktów końcowych palców
    num_fingers = 0

    # Sprawdzanie kciuka
    if hand_landmarks.landmark[3].y < hand_landmarks.landmark[2].y:
        num_fingers += 1

    # Sprawdzanie pozostałych palców
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 1].y and hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            num_fingers += 1

    return num_fingers - 2
# Definiowanie funkcji zwrotnej dla rozpoznawania gestów
detectN = 'None'
num_fingers = 0
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global detectN, num_fingers
    if result.gestures:
        detectN = result.gestures[0][0].category_name
        print('gesture recognition result:', result.gestures)
        handle_gesture(detectN, num_fingers)

# Tworzenie instancji rozpoznawania gestów
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_buffer=model_asset_buffer),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)
recognizer = GestureRecognizer.create_from_options(options)

# Inicjalizacja kamery
webcam = cv2.VideoCapture(0)

# Inicjalizacja sledzenai rak
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

#Petla główna
while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    image_height, image_width, _ = frame.shape
    
    # Konwertowanie obrazu z BGR na RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rozpoznawanie gestów
    mp_frame = mp_image(image_format=mp_image_format.SRGB, data=frame_rgb)
    recognizer.recognize_async(mp_frame, int(time.time() * 1000))

    # sledzenie rak
    results = hands.process(frame_rgb)

    # Rysowanie punktów charakterystycznych na rekach
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            num_fingers = count_fingers(hand_landmarks)
            for id, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                if id == 8:
                    cv2.circle(frame, (x + 50, y - 100), 8, (0, 255, 255), 2)
                if id == 4:
                    cv2.circle(frame, (x + 50, y - 100), 8, (0, 0, 255), 2)

    # Wyświetanie wyniku
    if detectN == 'None' and num_fingers in [3, 4]:
        cv2.putText(frame, f'Fingers: {num_fingers}', (10, 70), font, fontScale, fontColor, thickness, lineType)
    cv2.putText(frame, detectN, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
    cv2.imshow("CameraNTM", frame)

    key = cv2.waitKey(1)
    if key == ord('r'):
        break