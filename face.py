import mediapipe as mp
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf

class_names = ['Nastya', 'namemo', 'Masha', 'Tanya', 'Misha', 'Ksenia']

# Обращение к модулю обнаружения лиц в библиотеке MediaPipe
mp_face_detection = mp.solutions.face_detection
# Создание объекта обнаружения лиц
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

# Получение видеопотока с камеры
cap = cv2.VideoCapture(0)

model = load_model('./face_model2.h5')

iii = 0

idSot = {'Nastya': 0, 
      'namemo': 0, 
      'Masha': 0, 
      'Tanya': 0, 
      'Misha': 0, 
      'Ksenia': 0}

maxp = 0

diag = []

while True:

    success, image = cap.read()
    if not success:
        break

    # Преобразование цвета из BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Обработка кадра моделью для обнаружения лиц
    results = face_detection.process(image_rgb)

    cv2.imshow('Face Detection', image)
    d = image

    if results.detections is not None:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            
            left=bbox[0]
            top=bbox[1]
            right = bbox[0]+bbox[2]
            bottom = bbox[1]+bbox[3]

            img_new = d[top:bottom, left:right]

            if img_new is not None:
                cv2.imshow('', img_new) 

            input_data = cv2.resize(img_new, (640, 640))

            img_array = tf.keras.utils.img_to_array(input_data)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            if(abs(100 * np.max(score)) > 10):
                #print("На изображении скорее всего {} ({:.2f}% вероятность)".format(
                #    class_names[np.argmax(score)], 
                #    100 * np.max(score)))
                idSot[class_names[np.argmax(score)]] += 1

    iii += 1

    #if cv2.waitKey(1) & 0xFF == ord('q'):
    if iii == 10:
        for k, v in idSot.items():
            if maxp < v:
                maxp = v
                mode_price = k
        print("На изображении скорее всего ", mode_price)
        #if mode_price == "Nastya"^
        break

cap.release()
cv2.destroyAllWindows()
face_detection.close()

