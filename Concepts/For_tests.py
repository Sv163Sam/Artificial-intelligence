# print("ЭТО: ", np.array(model.support_vectors_).shape)

"""
def knn_train():
    X_train, X_test, y_train, y_test = train_test_split(result, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Создание модели KNN
    knn = KNeighborsClassifier(n_neighbors=5)

    # Обучение модели
    knn.fit(X_train, y_train)

    # Предсказание для тестовой выборки
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Вывод результатов
    print("\nРезультаты классификации:\n")

    print(f'Точность: {accuracy}\n')
    print(f'Матрица путаницы:\n{cm}\n\n')

    for i in range(len(X_test)):
        predicted_class = class_names[y_pred[i]]
        true_class = class_names[y_test[i]]

        # print(f"Строка {i + 1}: Предсказание - {predicted_class}, Правильный класс - {true_class}")


def decision_tree_train():
    X_train, X_test, y_train, y_test = train_test_split(result, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Создание модели Decision Tree
    dtc = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)

    # Обучение модели
    dtc.fit(X_train, y_train)

    # Предсказание для тестовой выборки
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Вывод результатов
    print("\nРезультаты классификации:\n")

    print(f'Точность: {accuracy}\n')
    print(f'Матрица путаницы:\n{cm}\n\n')

    for i in range(len(X_test)):
        predicted_class = class_names[y_pred[i]]
        true_class = class_names[y_test[i]]

        # print(f"Строка {i + 1}: Предсказание - {predicted_class}, Правильный класс - {true_class}")


def random_forest_train():
    X_train, X_test, y_train, y_test = train_test_split(result, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Создание модели Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # Обучение модели
    rf.fit(X_train, y_train)

    # Предсказание для тестовой выборки
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Вывод результатов
    print("\nРезультаты классификации:\n")

    print(f'Точность: {accuracy}\n')
    print(f'Матрица путаницы:\n{cm}\n\n')

    for i in range(len(X_test)):
        predicted_class = class_names[y_pred[i]]
        true_class = class_names[y_test[i]]

        print(f"Строка {i + 1}: Предсказание - {predicted_class}, Правильный класс - {true_class}")

"""
















"""
list = []

for i in X_train:
    list.append([np.mean(i), np.var(i)])

list = np.array(list)
plt.scatter(list[:, 0], list[:, 1])
plt.show()
"""




"""
import cv2
import mediapipe as mp

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Подключение к видеофайлу
cap = cv2.VideoCapture('path_to_video_file')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Обработка кадра MediaPipe Pose
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Получение координат скелета
    if results.pose_landmarks is not None:
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Отображение кадра с выделенной скелетной моделью
    cv2.imshow("Skeleton Model", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

"""
import cv2
import numpy as np
import mediapipe as mp
import time
import os

# Подключаем камеру
cap = cv2.VideoCapture(0)
cap.set(3, 640) # Width
cap.set(4, 480) # Lenght
cap.set(10, 100) # Brightness

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

pTime = 0
cTime = 0

#Зацикливаем получение кадров от камеры
while True:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        ret, frame = cap.read()
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)),(10,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2) # ФреймРейт

        cv2.imshow('python', image)
        
    if cv2.waitKey(1) == 27: # exit on ESC
        break
        
cv2.destroyWindow("python")
cap.release()
cv2.waitKey(1)
"""

"""
import cv2
import numpy as np
import mediapipe as mp
import time
import os

# Подключаем камеру
cap = cv2.VideoCapture(0) 
cap.set(3, 640) # Width
cap.set(4, 480) # Lenght
cap.set(10, 100) # Brightness

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
npDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
	
	#Зацикливаем получение кадров от камеры
while True: 
    success, img = cap.read()
    img = cv2.flip(img,1) # Mirror flip

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
               # print(id, lm)
                if  id == 8 or id == 12:
                    cv2.circle(img, (cx,cy),10,(255,0,255),cv2.FILLED)
            
            npDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(10,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2) # ФреймРейт
    
    cv2.imshow('python', img)
    if cv2.waitKey(20) == 27: # exit on ESC
        break
        
cv2.destroyWindow("python")
cap.release()
cv2.waitKey(1)
"""