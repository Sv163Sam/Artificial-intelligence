import os


def find_centers(source_file: str, output_file: str) -> None:
    """
    Находит центр масс человека по ключевым точкам после их детектирвоания

    Args:
        source_file (str): Имя файла с позами человека.
        output_file (str): Имя файла для записи результатов.
    Returns:
          None.
    """
    with open(source_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_number, line in enumerate(infile, 1):
            # Отбрасываем номер строки
            numbers = [float(x) for x in line.split()[1:]]

            # Вычисляем среднее значение для элементов 22, 24, 46, 48
            avg1 = sum(numbers[22:23] + numbers[24:25] + numbers[46:47] + numbers[48:49]) / 4

            # Вычисляем среднее значение для элементов 23, 25, 47, 49
            avg2 = sum(numbers[23:24] + numbers[25:26] + numbers[47:48] + numbers[49:50]) / 4

            # Записываем результат в новый файл
            outfile.write(f"{line_number}: {avg1:.4f}, {avg2:.4f}\n")


# Расчет новых координат
def translate_coordinates_to_centers(poses_path: str, centers_path: str, output_path: str, video_base_size: int):
    # Убедимся, что выходная папка существует
    os.makedirs(output_path, exist_ok=True)

    # Проходим по всем файлам в папке Poses
    for i in range(1, video_base_size + 1):
        input_filename = os.path.join(poses_path, f"{i}.txt")
        center_filename = os.path.join(centers_path, f"{i}.txt")
        output_filename = os.path.join(output_path, f"{i}.txt")

        with open(input_filename, 'r') as poses, open(center_filename, 'r') as centers, open(output_filename, 'w') as output:
            center_lines = centers.readlines()  # Читаем центр масс
            centers = [list(map(float, line.split(':')[1].strip().split(', '))) for line in center_lines]

            for line_number, line in enumerate(poses, 1):
                # Отбрасываем номер строки
                pose_values = list(map(float, line.split()[1:]))

                # Получаем центр масс для текущего кадра
                center_x, center_y = centers[line_number - 1]

                # Вычисляем новые координаты
                translated_coordinates = []
                for j in range(0, len(pose_values), 2):
                    x = pose_values[j] - center_x
                    y = pose_values[j + 1] - center_y
                    translated_coordinates.append((x, y))

                # Записываем результат в новый файл
                output.write(f"{line_number}: ")
                output.write(' '.join(f"{x:.4f} {y:.4f}" for x, y in translated_coordinates))
                output.write("\n")