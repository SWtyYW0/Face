import numpy as np
import cv2

# Используемые константы
CASCADE_FILE_PATH = './haarcascade_frontalface_alt.xml'
DATASET_PATH = './face_dataset/'

# Инициализация видеопотока
cap = cv2.VideoCapture(0)

# Инициализация каскада Хаара
face_cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)

# Переменные для сбора данных о лицах
skip_frames = 0
face_data = []

# Запрос имени пользователя
file_name = input("Enter your name: ")

while True:
    # Захват кадра из видеопотока
    ret, frame = cap.read()
    if not ret:
        continue

    # Преобразование кадра в оттенки серого
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    # Обработка найденных лиц
    for face in faces[:1]:
        x, y, w, h = face

        # Обрезка области с лицом и изменение размера
        offset = 7
        face_section = frame[y - offset : y + h + offset, x - offset : x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Сбор данных о лице каждый 10-й кадр
        if skip_frames % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))

        # Отображение лица в отдельном окне
        cv2.imshow(str(skip_frames), face_section)

        # Отрисовка прямоугольника вокруг лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Увеличение счетчика пропущенных кадров
    skip_frames += 1

    # Отображение кадра с лицами
    cv2.imshow("Faces", frame)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Преобразование данных о лицах в массив NumPy
face_data = np.asarray(face_data)

# Вывод формы массива и его сохранение
print(face_data.shape)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

np.save(DATASET_PATH + file_name, face_data)
print("Dataset saved at: {}".format(DATASET_PATH + file_name + '.npy'))

# Завершение работы с видеопотоком и закрытие окон
cap.release()
cv2.destroyAllWindows()
