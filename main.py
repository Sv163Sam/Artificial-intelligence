from Analyzators.Processing import *
from Analyzators.Dot_Finding import *
from Analyzators.Mass_center import *
from Utils.Project_utils import *


import os


if __name__ == "__main__":
    print("РАСПОЗНАВАНИЕ ДЕЙСТВИЙ ЧЕЛОВЕКА В ОХРАНЯЕМОЙ ЗОНЕ\n")

    while True:
        video_base_size = count_video_files(SOURCE_PATH)
        count_frames_in_files = []

        print("Выберите действие: "
              "\n1 - Детектирование позы человека на видеофрагменте"
              "\n2 - Детектирование центра масс для каждого видеофрагмента"
              "\n3 - Классификация действий человека"
              "\n4 - Выйти")
        user_input = input("")

        if user_input == "1":
            print(f"Вызов функции pose_detection с параметрами: размер базы данных видеозаписей - {video_base_size}")
            pose_detection(video_base_size, SOURCE_PATH, RESULT_POSES_PATH)
            print(f"Позы каждого видеофрагмента определены в директории: {RESULT_POSES_PATH}")
            continue
        if user_input == "2":
            print(f"Вызов функции find_centers")
            for i in range(video_base_size):
                source_filename = os.path.join(RESULT_POSES_PATH, f"{str(i + 1)}.txt")
                output_filename = os.path.join(RESULT_CENTERS_PATH, f"{str(i + 1)}.txt")
                find_centers(source_filename, output_filename)
            continue
        if user_input == "3":
            count_frames_in_files = count_frames(video_base_size, RESULT_POSES_PATH)

            # Вызовы
            print(f"Вызов функции translate_coordinates_to_centers с параметрами: размер базы данных видеозаписей - {video_base_size}")
            translate_coordinates_to_centers(RESULT_POSES_PATH, RESULT_CENTERS_PATH, RESULT_TRANSLATED_CENTERS_PATH,
                                             video_base_size)

            # Значения из файла смещения центров
            print(f"Вызов функции read_files с параметрами: размер базы данных видеозаписей - {video_base_size}")
            translated_centers_values = read_files(video_base_size, RESULT_TRANSLATED_CENTERS_PATH)

            # Результирующая матрица, в которой хранится каждый проход окно в виде строки: list[array1, array2, ...]
            print(f"Вызов функции window_processing с параметрами: размер окна - 10, шаг окна - 2")
            window_processed_data = window_processing(translated_centers_values, count_frames_in_files, 10, 2)

            # Метки
            labels = np.zeros(len(window_processed_data), dtype=int)

            # Вектор меток для обучения
            labels_names = ["Ходьба", "Бег", "Пройти через турникет", "Перешагнуть через ограждение",
                            "Пролезть сквозь ограждение", "Кинуть предмет", "Бросить сумку", "Постучать и заглянуть",
                            "Постучать и зайти"]

            # Инициализация меток в соответствие с количеством видеозаписей
            labels[0:int(count_frames_in_files[0] / 2) + 1] = 0
            start_index = int(count_frames_in_files[0] / 2) + 1
            for i in range(1, 19):
                labels[start_index: start_index + int(count_frames_in_files[i] / 2) + 1] = 0
                start_index += int(count_frames_in_files[i] / 2) + 1
            for i in range(19, 37):
                labels[start_index: start_index + int(count_frames_in_files[i] / 2) + 1] = 1
                start_index += int(count_frames_in_files[i] / 2) + 1
            for i in range(37, 57):
                labels[start_index: start_index + int(count_frames_in_files[i] / 2) + 1] = 2
                start_index += int(count_frames_in_files[i] / 2) + 1
            for i in range(57, 77):
                labels[start_index: start_index + int(count_frames_in_files[i] / 2) + 1] = 3
                start_index += int(count_frames_in_files[i] / 2) + 1
            for i in range(77, 97):
                labels[start_index: start_index + int(count_frames_in_files[i] / 2) + 1] = 4
                start_index += int(count_frames_in_files[i] / 2) + 1
            for i in range(97, 115):
                labels[start_index: start_index + int(count_frames_in_files[i] / 2) + 1] = 5
                start_index += int(count_frames_in_files[i] / 2) + 1
            for i in range(115, 131):
                labels[start_index: start_index + int(count_frames_in_files[i] / 2) + 1] = 6
                start_index += int(count_frames_in_files[i] / 2) + 1
            for i in range(131, 141):
                labels[start_index: start_index + int(count_frames_in_files[i] / 2) + 1] = 7
                start_index += int(count_frames_in_files[i] / 2) + 1
            for i in range(141, 159):
                labels[start_index: start_index + int(count_frames_in_files[i] / 2) + 1] = 8
                start_index += int(count_frames_in_files[i] / 2) + 1

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
                            not_mixed_samples_train(window_processed_data, labels, labels_names, "SVC",
                                                    count_frames_in_files)
                            break
                        if classifier_input == "2":
                            print(
                                f"Вызов функции window_processing с параметрами: классификатор - KNN")
                            not_mixed_samples_train(window_processed_data, labels, labels_names, "KNN",
                                                    count_frames_in_files)
                            break
                        if classifier_input == "3":
                            print(
                                f"Вызов функции window_processing с параметрами: классификатор - DecisionTree")
                            not_mixed_samples_train(window_processed_data, labels, labels_names, "DecisionTree",
                                                    count_frames_in_files)
                            break
                        if classifier_input == "4":
                            print(
                                f"Вызов функции window_processing с параметрами: классификатор - RandomForest")
                            not_mixed_samples_train(window_processed_data, labels, labels_names, "RandomForest",
                                                    count_frames_in_files)
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
