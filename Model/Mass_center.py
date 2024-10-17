N = 89


def find_centrs(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line_number, line in enumerate(infile, 1):
            # Разделяем строку на числа
            numbers = [float(x) for x in line.split()[1:]]  # Отбрасываем номер строки

            # Вычисляем среднее значение для элементов 22, 24, 46, 48
            avg1 = sum(numbers[22:23] + numbers[24:25] + numbers[46:47] + numbers[48:49]) / 4

            # Вычисляем среднее значение для элементов 23, 25, 47, 49
            avg2 = sum(numbers[23:24] + numbers[25:26] + numbers[47:48] + numbers[49:50]) / 4

            # Записываем результат в новый файл
            outfile.write(f"{line_number}: {avg1:.4f}, {avg2:.4f}\n")


# Пример использования:
for i in range(N):
    input_filename = "../Poses/" + str(i + 1) + ".txt"
    output_filename = "../Centrs/" + str(i + 1) + ".txt"
    find_centrs(input_filename, output_filename)
