import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os


# Чтение из файлов с описанием видеозаписей в один общий массив numbers
def read_files(video_base_size: int) -> list[list[float]]:
    """
    Читает построчно файл + парсит номера строка

    Args:
        video_base_size (int): Размер базы данных с видеозаписями.
    Returns:
        list[list[float]] - Сформированная матрица из всех файлов (последовательно).
    """
    file_values = []

    for file_index in range(0, video_base_size):
        filename = f'Processing_files/Translated_Centers/{file_index + 1}.txt'

        with open(filename, 'r') as file:
            data = file.readlines()
        for line in data:
            file_values.append([float(num) for num in line.split()[1:]])

    return file_values


# Создание общей матрицы, в которой хранится обработка скользящим окном по массиву numbers
def window_processing(file_values: list, window_size: int = 10, window_step: int = 2) -> list[np.array]:
    """
    Обработка скользящим окном

    Args:
        file_values (list): Сформированная матрица из файлов.
        window_size (int): Размер окна.
        window_step (int): Шаг окна.
    Returns:
        list[np.array] - Сформированная матрица в результате обработки скользящим окном.
    """
    result = []
    for i in range(0, len(file_values) + 1 - window_size, window_step):
        if (i // 40) * 40 + 30 < i < (i // 40 + 1) * 40 + 40:
            continue

        window_matrix = np.array(file_values[i:i + window_size])
        result.append(window_matrix.flatten())

    return result


# Обучение на тренировочной выборке и предсказание на тестовой с помощью: SVC, KNN, DecisionTree, RandomForest
# Данные не обрабатываются предварительно, то есть описание одной видеозаписи попадает в обе выборки
def mixed_samples_train(window_processed_data: list[np.array], labels: np.array, labels_names: list[str], classifier_type: str):
    """
    Классификация на выборках без предварительной обработки.

    Args:
        window_processed_data (list[np.array]): Сформированная в результате обработки скользящим окном матрица.
        labels (np.array): Массив меток.
        labels_names (list[str): Расшифровка меток.
        classifier_type (str): Тип классификатора.
    Returns:
        None.
    """
    scaler = StandardScaler()
    scaler.fit(window_processed_data)

    # Разделение на выборки
    x_train, x_test, y_train, y_test = train_test_split(window_processed_data, labels, test_size=0.5, random_state=42)

    # Перемешивание
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Обучение
    def svc():
        print("Модель SVC без предварительной обработки")
        svc_model = SVC(kernel='rbf', C=1)
        svc_model.fit(x_train, y_train)
        return svc_model

    def knn():
        print("Модель KNN без предварительной обработки")
        kn_model = KNeighborsClassifier(n_neighbors=5)
        kn_model.fit(x_train, y_train)
        return kn_model

    def decision_tree():
        print("Модель DecisionTree без предварительной обработки")
        df_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
        df_model.fit(x_train, y_train)
        return df_model

    def random_forest():
        print("Модель RandomForest без предварительной обработки")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(x_train, y_train)
        return rf_model

    model = None
    if classifier_type == "SVC":
        model = svc()
    if classifier_type == "KNN":
        model = knn()
    if classifier_type == "DecisionTree":
        model = decision_tree()
    if classifier_type == "RandomForest":
        model = random_forest()

    # Предсказание
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Вывод результатов
    print(f'Точность: {accuracy}\n')
    print(f'Матрица путаницы:\n{cm}\n\n')

    for i in range(len(x_test)):
        predicted_class = labels_names[y_pred[i]]
        true_class = labels_names[y_test[i]]

        # print(f"Окно {i + 1}: Предсказание - {predicted_class}, Правильный класс - {true_class}")


# Обучение на тренировочной выборке и предсказание на тестовой с помощью: SVC, KNN, DecisionTree, RandomForest
# Данные обрабатываются предварительно(группируются), то есть описание одной видеозаписи принадлежит только одной из выборок
def not_mixed_samples_train(window_processed_data: list[np.array], labels: np.array, labels_names: list[str], classifier_type: str):
    """
    Классификация на выборках с предварительной обработкой.

    Args:
        window_processed_data (list[np.array]): Сформированная в результате обработки скользящим окном матрица.
        labels (np.array): Массив меток.
        labels_names (list[str): Расшифровка меток.
        classifier_type (str): Тип классификатора.
    Returns:
        None.
    """
    # Количество групп
    n_entities = len(window_processed_data) // 16

    # Группировка
    grouped_result = [window_processed_data[i * 16:(i + 1) * 16] for i in range(n_entities)]
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
        print("Модель SVC с предварительной обработкой")
        svc_model = SVC(kernel='rbf', C=1)
        svc_model.fit(x_train, y_train)
        return svc_model

    def knn():
        print("Модель KNN с предварительной обработкой")
        kn_model = KNeighborsClassifier(n_neighbors=5)
        kn_model.fit(x_train, y_train)
        return kn_model

    def decision_tree():
        print("Модель DecisionTree с предварительной обработкой")
        df_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
        df_model.fit(x_train, y_train)
        return df_model

    def random_forest():
        print("Модель RandomForest с предварительной обработкой")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(x_train, y_train)
        return rf_model

    model = None
    if classifier_type == "SVC":
        model = svc()
    if classifier_type == "KNN":
        model = knn()
    if classifier_type == "DecisionTree":
        model = decision_tree()
    if classifier_type == "RandomForest":
        model = random_forest()

    # Предсказание
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Вывод результатов
    print(f'Точность: {accuracy}\n')
    print(f'Матрица путаницы:\n{cm}\n\n')

    for i in range(len(x_test)):
        predicted_class = labels_names[y_pred[i]]
        true_class = labels_names[y_test[i]]

        # print(f"Строка {i + 1}: Предсказание - {predicted_class}, Правильный класс - {true_class}")

