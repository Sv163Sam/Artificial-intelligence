from Model.Processing import *
from Model.Dot_Finding import pose_detection
from Model.Mass_center import *

if __name__ == "__main__":
    print("РАСПОЗНАВАНИЕ ДЕЙСТВИЙ ЧЕЛОВЕКА В ОХРАНЯЕМОЙ ЗОНЕ\n")

    try:
        while True:
            video_base_size = 89

            print("Выберите действие: "
                  "\n1 - Детектирование позы человека на видеофрагменте"
                  "\n2 - Детектирование центра масс для каждого видеофрагмента"
                  "\n3 - Классификация действий человека"
                  "\n4 - Выйти")
            user_input = input("")

            if user_input == "1":
                count_frames = 40
                print(
                    f"Вызов функции pose_detection с параметрами: размер базы данных видеозаписей - {video_base_size}, количество кадров для обработки - {count_frames}")
                pose_detection(video_base_size, count_frames)
                print("Позы каждого видеофрагмента определены в директории Processing_files/Poses")
                pass
            if user_input == "2":
                print(f"Вызов функции find_centers")
                for i in range(video_base_size):
                    source_filename = "Processing_files/Poses/" + str(i + 1) + ".txt"
                    output_filename = "Processing_files/Centers/" + str(i + 1) + ".txt"
                    find_centers(source_filename, output_filename)
                pass
            if user_input == "3":
                # Метки
                labels = np.zeros(video_base_size * 16, dtype=int)
                # Вектор меток для обучения
                labels_names = ["Бег(боковая проекция)", "Присяд", "Выпады", "Наклон туловища",
                                "Ходьба(фронтальная проекция)", "Ходьба(боковая проекция)", "Вертикальный прыжок"]

                # Вызовы
                print(
                    f"Вызов функции translate_coordinates_to_centers с параметрами: размер базы данных видеозаписей - {video_base_size}")
                translate_coordinates_to_centers("Processing_files/Poses", "Processing_files/Centers",
                                                 "Processing_files/Translated_Centers", video_base_size)

                # Значения из файла смещения центров
                print(f"Вызов функции read_files с параметрами: размер базы данных видеозаписей - {video_base_size}")
                translated_centers_values = read_files(video_base_size)

                # Результирующая матрица, в которой хранится каждый проход окно в виде строки: list[array1, array2, ...]
                print(f"Вызов функции window_processing с параметрами: размер окна - 10, шаг окна - 2")
                window_processed_data = window_processing(translated_centers_values, 10, 2)

                # Инициализация меток в соответствие с количеством видеозаписей
                labels[0:160] = 0
                labels[160:320] = 1
                labels[320:480] = 2
                labels[480:640] = 3
                labels[640:960] = 4
                labels[960:1120] = 5
                labels[1120:] = 6

                while True:
                    print("Выберите тип классификации:"
                          "\n1 - С предварительной обработкой"
                          "\n2 - Без предварительной обработки"
                          "\n3 - Выйти")
                    classification_input = input("")
                    if classification_input == "1":
                        while True:
                            print("Выберите тип классификатора:"
                                  "\n1 - SVC"
                                  "\n2 - KNN"
                                  "\n3 - DecisionTree"
                                  "\n4 - RandomForest")
                            classifier_input = input("")
                            if classifier_input == "1":
                                print(
                                    f"Вызов функции window_processing с параметрами: классификатор - SVC")
                                not_mixed_samples_train(window_processed_data, labels, labels_names, "SVC")
                                break
                            if classifier_input == "2":
                                print(
                                    f"Вызов функции window_processing с параметрами: классификатор - KNN")
                                not_mixed_samples_train(window_processed_data, labels, labels_names, "KNN")
                                break
                            if classifier_input == "3":
                                print(
                                    f"Вызов функции window_processing с параметрами: классификатор - DecisionTree")
                                not_mixed_samples_train(window_processed_data, labels, labels_names, "DecisionTree")
                                break
                            if classifier_input == "4":
                                print(
                                    f"Вызов функции window_processing с параметрами: классификатор - RandomForest")
                                not_mixed_samples_train(window_processed_data, labels, labels_names, "RandomForest")
                                break
                            else:
                                pass
                        break
                    if classification_input == "2":
                        while True:
                            print("Выберите тип классификатора:"
                                  "\n1 - SVC"
                                  "\n2 - KNN"
                                  "\n3 - DecisionTree"
                                  "\n4 - RandomForest")
                            classifier_input = input("")
                            if classifier_input == "1":
                                print(
                                    f"Вызов функции window_processing с параметрами: классификатор - SVC")
                                mixed_samples_train(window_processed_data, labels, labels_names, "SVC")
                                break
                            if classifier_input == "2":
                                print(
                                    f"Вызов функции window_processing с параметрами: классификатор - KNN")
                                mixed_samples_train(window_processed_data, labels, labels_names, "KNN")
                                break
                            if classifier_input == "3":
                                print(
                                    f"Вызов функции window_processing с параметрами: классификатор - DecisionTree")
                                mixed_samples_train(window_processed_data, labels, labels_names, "DecisionTree")
                                break
                            if classifier_input == "4":
                                print(f"Вызов функции window_processing с параметрами: классификатор - RandomForest")
                                mixed_samples_train(window_processed_data, labels, labels_names, "RandomForest")
                                break
                        break
                    if classification_input == "3":
                        break
                    else:
                        pass
                pass
            else:
                break
    except Exception as e:
        print(e)
        exit(1)
