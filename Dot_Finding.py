import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_array = ["NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
              "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
              "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
              "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
              "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]


for i in range(95):  # Пробегаем по исходникам
    iteration = 0
    strin = "Source/" + str(i + 1) + ".mp4"
    cap = cv2.VideoCapture(strin)  # Перадаем исходники в нейронку

    filename = "Resultxt/" + str(i + 1) + ".txt"  # Задаем имя файл пропорционально исходнику

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and iteration < 40:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Открывает файл на дозапись
            # Запишем туда номер фрейма и координаты

            with open(filename, 'a') as file:
                file.write("\n")
                file.write(str(iteration + 1) + ': ')
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    # print(f"Landmark {mp_pose.PoseLandmark(i).name}: {landmark.x}, {landmark.y}, {landmark.z}")
                    file.write(str(landmark.x) + ' ')
                    file.write(str(landmark.y) + ' ')

            file.close()
            iteration = iteration + 1

            # print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # cv2.imshow('Video', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
