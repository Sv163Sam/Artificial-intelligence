import cv2
import os
import mediapipe as mp


def pose_detection(video_base_size: int, source_path: str, destination_path: str) -> None:
    """
    Определяет координаты ключевых точек на видеофрагменте

    Args:
        video_base_size (int): Количество видеозаписей в директории.
        source_path (str): Путь к видеозаписям.
        destination_path (str): Путь, куда будут записываться позы.
    Returns:
        None.
    """
    # Инициализация методов библиотеки
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Определение поз
    for i in range(video_base_size):
        frames = 0   # Счетчик кадров для записи в файл

        # Чтение каждой видеозаписи и инициализация переменной для обработки
        source_filename = os.path.join(source_path, f"{str(i + 1)}.mov")

        # Инициализация названия файла на запись результата
        destination_filename = os.path.join(destination_path, f"{str(i+1)}.txt")

        cap = cv2.VideoCapture(source_filename)

        # Детектирование поз при помощи скелетной модели и запись в файл
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Подготовка
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Детектирование
                results = pose.process(image)

                # Обратное преобразование после использования в детектировании
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Запись в файл
                with open(destination_filename, 'a') as file:
                    file.write(str(frames + 1) + ': ')  # Запись номера кадра в файл

                    if results.pose_landmarks is not None:
                        # Запись координат
                        for inc, landmark in enumerate(results.pose_landmarks.landmark):
                            file.write(str(landmark.x) + ' ')
                            file.write(str(landmark.y) + ' ')
                    else:
                        # Если landmarks отсутствуют, записываем нули
                        for _ in range(len(mp_pose.PoseLandmark)):
                            file.write('0.0 0.0 ')

                    file.write("\n")  # Новая строка

                frames += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()

    cv2.destroyAllWindows()

