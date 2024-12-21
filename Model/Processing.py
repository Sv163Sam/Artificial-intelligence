import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os

# Параметры
N = 89  # Количество файлов
T = 10  # Размер окна
S = 2  # Шаг окна

# Расчет новых координат
def translate_coordinates_to_centers(input_folder_poses, input_folder_centers, output_folder):
    # Убедимся, что выходная папка существует
    os.makedirs(output_folder, exist_ok=True)

    # Проходим по всем файлам в папке Poses
    for i in range(1, N + 1):  # Замените N на количество файлов
        input_filename = os.path.join(input_folder_poses, f"{i}.txt")
        center_filename = os.path.join(input_folder_centers, f"{i}.txt")
        output_filename = os.path.join(output_folder, f"{i}.txt")

        with open(input_filename, 'r') as infile, open(center_filename, 'r') as centerfile, open(output_filename,
                                                                                                 'w') as outfile:
            # Читаем центр масс
            center_lines = centerfile.readlines()
            centers = [list(map(float, line.split(':')[1].strip().split(', '))) for line in center_lines]

            # Обрабатываем каждую строку с позами
            for line_number, line in enumerate(infile, 1):
                # Разделяем строку на числа
                numbers = list(map(float, line.split()[1:]))  # Отбрасываем номер строки

                # Получаем центр масс для текущего кадра
                center_x, center_y = centers[line_number - 1]

                # Вычисляем новые координаты
                translated_coordinates = []
                for j in range(0, len(numbers), 2):
                    x = numbers[j] - center_x
                    y = numbers[j + 1] - center_y
                    translated_coordinates.append((x, y))

                # Записываем результат в новый файл
                outfile.write(f"{line_number}: ")
                outfile.write(' '.join(f"{x:.4f} {y:.4f}" for x, y in translated_coordinates))
                outfile.write("\n")


# Контейнеры
numbers = []  # Для чтения из файла
result = []  # Результирующая матрица, в которой хранится каждый проход окно в виде строки: list[array1, array2, ...]

# Метки
labels = np.zeros(1424, dtype=int)  # Вектор меток для обучения
class_names = ["Бег(боковая проекция)", "Присяд", "Выпады", "Наклон туловища", "Ходьба(фронтальная проекция)",
               "Ходьба(боковая проекция)", "Вертикальный прыжок"]  # Названия меток для вывода

# Индексы меток
Run_side = 0  # Бег(боковая проекция)
Sitdown = 1  # Присяд
Lunges = 2  # Выпады
Torso_tilt = 3  # Наклон туловища
Walk_front = 4  # Ходьба(фронтальная проекция)
Walk_side = 5  # Ходьба(боковая проекция)
Jump = 6  # Вертикальный прыжок


# Чтение из файлов с описанием видеозаписей в один общий массив numbers
def read_files():
    for file_index in range(0, N):
        filename = f'../Translated_Centers/{file_index + 1}.txt'
        with open(filename, 'r') as file:
            data = file.readlines()
        for line in data:
            numbers.append([float(num) for num in line.split()[1:]])


# Вывести массив numbers
def print_numbers():
    for line in numbers:
        print(line)


# Вывести массив result
def print_result():
    print(result)


# Создание общей матрицы, в которой хранится обработка скользящим окном по массиву numbers
def create_res():
    for i in range(0, len(numbers) + 1 - T, S):
        if (i // 40) * 40 + 30 < i < (i // 40 + 1) * 40 + 40:
            continue
        window_matrix = np.array(numbers[i:i + T])
        result.append(window_matrix.flatten())


# Инициализация меток в соответствие с количеством видеозаписей
def init_labels():
    labels[0:160] = Run_side
    labels[160:320] = Sitdown
    labels[320:480] = Lunges
    labels[480:640] = Torso_tilt
    labels[640:960] = Walk_front
    labels[960:1120] = Walk_side
    labels[1120:] = Jump


# Обучение на тренировочной выборке и предсказание на тестовой с помощью: SVC, KNN, DecisionTree, RandomForest
# Данные не обрабатываются предварительно, то есть описание одной видеозаписи попадает в обе выборки
def mixed_samples_train():
    scaler = StandardScaler()
    scaler.fit(result)

    # Разделение на выборки
    x_train, x_test, y_train, y_test = train_test_split(result, labels, test_size=0.5, random_state=42)

    # Перемешивание
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Обучение
    def svc():
        print("Модель SVC(Mixed):")
        svc_model = SVC(kernel='rbf', C=1)
        svc_model.fit(x_train, y_train)
        return svc_model

    def kn():
        print("Модель KN(Mixed):")
        kn_model = KNeighborsClassifier(n_neighbors=5)
        kn_model.fit(x_train, y_train)
        return kn_model

    def df():
        print("Модель DF(Mixed):")
        df_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
        df_model.fit(x_train, y_train)
        return df_model

    def rf():
        print("Модель RF(Mixed):")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(x_train, y_train)
        return rf_model

    model = svc()
    # model = kn()
    # model = df()
    # model = rf()

    # Предсказание
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Вывод результатов
    print(f'Точность: {accuracy}\n')
    print(f'Матрица путаницы:\n{cm}\n\n')

    for i in range(len(x_test)):
        predicted_class = class_names[y_pred[i]]
        true_class = class_names[y_test[i]]

        # print(f"Окно {i + 1}: Предсказание - {predicted_class}, Правильный класс - {true_class}")


# Обучение на тренировочной выборке и предсказание на тестовой с помощью: SVC, KNN, DecisionTree, RandomForest
# Данные обрабатываются предварительно(группируются), то есть описание одной видеозаписи принадлежит только одной из выборок
def not_mixed_samples_train():
    # Количество групп
    n_entities = len(result) // 16

    # Группировка
    grouped_result = [result[i * 16:(i + 1) * 16] for i in range(n_entities)]
    grouped_labels = [labels[i * 16:(i + 1) * 16] for i in range(n_entities)]

    # Разделение на обучающую и тестовую выборки
    train_indices, test_indices = train_test_split(list(range(n_entities)), test_size=0.2, random_state=42)

    # Создание обучающей и тестовой выборок
    x_train = [grouped_result[i] for i in train_indices]
    x_test = [grouped_result[i] for i in test_indices]
    y_train = [grouped_labels[i] for i in train_indices]
    y_test = [grouped_labels[i] for i in test_indices]

    # Развертывание данных
    x_train = [item for sublist in x_train for item in sublist]
    x_test = [item for sublist in x_test for item in sublist]
    y_train = [item for sublist in y_train for item in sublist]
    y_test = [item for sublist in y_test for item in sublist]

    # Перемешивание
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Обучение
    def svc():
        print("Модель SVC(Not mixed):")
        svc_model = SVC(kernel='rbf', C=1)
        svc_model.fit(x_train, y_train)
        return svc_model

    def kn():
        print("Модель KN(Not mixed):")
        kn_model = KNeighborsClassifier(n_neighbors=5)
        kn_model.fit(x_train, y_train)
        return kn_model

    def df():
        print("Модель DF(Not mixed):")
        df_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
        df_model.fit(x_train, y_train)
        return df_model

    def rf():
        print("Модель RF(Not mixed):")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(x_train, y_train)
        return rf_model

    model = svc()
    # model = kn()
    # model = df()
    # model = rf()

    # Предсказание
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Вывод результатов
    print(f'Точность: {accuracy}\n')
    print(f'Матрица путаницы:\n{cm}\n\n')

    for i in range(len(x_test)):
        predicted_class = class_names[y_pred[i]]
        true_class = class_names[y_test[i]]

        # print(f"Строка {i + 1}: Предсказание - {predicted_class}, Правильный класс - {true_class}")


# Вызовы
translate_coordinates_to_centers("../Poses", "../Centrs", "../Translated_Centers")
read_files()
create_res()
init_labels()
# mixed_samples_train()
not_mixed_samples_train()
