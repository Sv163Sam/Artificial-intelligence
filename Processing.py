import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Параметры
N = 89  # Количество файлов
T = 10  # Размер окна
S = 2  # Шаг окна

# Контейнеры
numbers = []  # Для чтения из файла
result = []  # Результирующая матрица, в которой хранится каждый проход окно в виде строки: list[array1, array2, ...]

# Метки
labels = np.zeros(1424, dtype=int)  # Вектор меток для обучения
class_names = ["Бег(боковая проекция)", "Присяд", "Выпады", "Наклон туловища", "Ходьба(фронтальная проекция)",
               "Вертикальный прыжок", "Ходьба(боковая проекция)"]  # Названия меток для вывода

# Индексы меток
Run_side = 0  # Бег(боковая проекция)
Sitdown = 1  # Присяд
Lunges = 2  # Выпады
Torso_tilt = 3  # Наклон туловища
Walk_front = 4  # Ходьба(фронтальная проекция)
Jump = 5  # Вертикальный прыжок
Walk_side = 6  # Ходьба(боковая проекция)


# Чтение из файлов с описанием видеозаписей в один общий массив numbers
def read_files():
    for file_index in range(0, N):
        filename = f'Resultxt/{file_index + 1}.txt'
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
        if any(30 + 40 * k < i < 40 + 40 * k for k in range(int((len(numbers) + 1 - T) / 40))):
            continue
        window_matrix = np.array(numbers[i:i + T])  # Получаем матрицу размера T * n
        flattened_row = window_matrix.flatten()  # Развернуть матрицу в строку
        result.append(flattened_row)  # Добавить строку в результат


# Инициализация меток в соответствие с количеством видеозаписей
def init_labels():
    labels[0:160] = Run_side
    labels[160:320] = Sitdown
    labels[320:480] = Lunges
    labels[480:640] = Torso_tilt
    labels[640:960] = Walk_front
    labels[960:1120] = Jump
    labels[1120:] = Walk_side


# Обучение на тренировочной выборке и предсказание на тестовой с помощью: SVC, KNN, DecisionTree, RandomForest
def mixed_samples_train():
    # Разделение на выборки
    x_train, x_test, y_train, y_test = train_test_split(result, labels, test_size=0.5, random_state=42)

    # Перемешивание
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Обучение
    def svc():
        svc_model = SVC(kernel='rbf', C=1)
        svc_model.fit(x_train, y_train)
        return svc_model

    def knn():
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(x_train, y_train)
        return knn_model

    def df():
        df_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
        df_model.fit(x_train, y_train)
        return df_model

    def rf():
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(x_train, y_train)
        return rf_model

    model = svc()
    # model = knn()
    # model = df()
    # model = rf()

    # Предсказание
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Вывод результатов
    print("\nРезультаты классификации:\n")

    print(f'Точность: {accuracy}\n')
    print(f'Матрица путаницы:\n{cm}\n\n')

    for i in range(len(x_test)):
        predicted_class = class_names[y_pred[i]]
        true_class = class_names[y_test[i]]

        # print(f"Окно {i + 1}: Предсказание - {predicted_class}, Правильный класс - {true_class}")


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
        svc_model = SVC(kernel='rbf', C=1)
        svc_model.fit(x_train, y_train)
        return svc_model

    def knn():
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(x_train, y_train)
        return knn_model

    def df():
        df_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
        df_model.fit(x_train, y_train)
        return df_model

    def rf():
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(x_train, y_train)
        return rf_model

    model = svc()
    # model = knn()
    # model = df()
    # model = rf()

    # Предсказание
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Вывод результатов
    print("\nРезультаты классификации:\n")

    print(f'Точность: {accuracy}\n')
    print(f'Матрица путаницы:\n{cm}\n\n')

    for i in range(len(x_test)):
        predicted_class = class_names[y_pred[i]]
        true_class = class_names[y_test[i]]

        # print(f"Строка {i + 1}: Предсказание - {predicted_class}, Правильный класс - {true_class}")


# Вызовы
read_files()
create_res()
init_labels()
# mixed_samples_train()
# not_mixed_samples_train()
