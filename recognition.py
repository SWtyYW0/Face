import numpy as np
import cv2
import os

# KNN 
def calculate_distance(v1, v2):
    # Евклидово расстояние
    return np.sqrt(((v1 - v2) ** 2).sum())

def k_nearest_neighbors(train_data, test_data, k=5):
    distances = []

    for i in range(train_data.shape[0]):
        # вектор и метка
        train_vector = train_data[i, :-1]
        train_label = train_data[i, -1]
        # расстояние от контрольной точки
        dist = calculate_distance(test_data, train_vector)
        distances.append([dist, train_label])

    # Сортировка по расстоянию и получение топ k
    k_nearest = sorted(distances, key=lambda x: x[0])[:k]
    # Извлекать только метки
    labels = np.array(k_nearest)[:, -1]

    # Частоты каждой метки
    output = np.unique(labels, return_counts=True)
    # Найдите максимальную частоту и соответствующую метку
    index = np.argmax(output[1])
    return output[0][index]


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
dataset_path = './face_dataset/'

face_data = []
labels = []
class_id = 0

for file_name in os.listdir(dataset_path):
    if file_name.endswith('.npy'):
        data_item = np.load(os.path.join(dataset_path, file_name))
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_labels.shape)
print(face_dataset.shape)

train_set = np.concatenate((face_dataset, face_labels), axis=1)
print(train_set.shape)

names = {
    0: 'Ikram',
    1: 'Ali',
    2: 'Dauren',
    3: 'Kazbek',
    4: 'Aizhan'
}

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if ret == False:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        x, y, w, h = face

        offset = 7
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        result = k_nearest_neighbors(train_set, face_section.flatten())

        cv2.putText(frame, names[int(result)], (x, y - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
