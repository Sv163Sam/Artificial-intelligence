import cv2
import mediapipe as mp


def pose_detection(video_base_size: int, frames_count: int = 40) -> None:
    """
    Определяет координаты ключевых точек на видеофрагменте

    Args:
        video_base_size (int): Количество видеозаписей в директории.
        frames_count (int): Количество обрабатываемых кадров в видеофрагменте
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
        source_filename = "../Processing_files/Source/" + str(i + 1) + ".mp4"
        # Инициализация названия файла на запись результата
        destination_filename = "../Processing_files/Poses/" + str(i + 1) + ".txt"

        cap = cv2.VideoCapture(source_filename)

        # Детектирование поз при помощи скелетной модели и запись в файл
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and frames < frames_count:
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
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Запись в файл
                with open(destination_filename, 'a') as file:
                    file.write(str(frames + 1) + ': ')  # Запись номера кадра в файл

                    # Запись координат
                    for inc, landmark in enumerate(results.pose_landmarks.landmark):
                        file.write(str(landmark.x) + ' ')
                        file.write(str(landmark.y) + ' ')

                    file.write("\n")  # Новая строка

                file.close()  # Завершение работы с файлом
                frames = frames + 1  # Инкремент кадра

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
