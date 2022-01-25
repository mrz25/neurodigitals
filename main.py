import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten


# from keras.models import load_model
# from tkinter import *
# import tkinter as tk
# import win32gui
# from PIL import ImageGrab, Image


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary())      # вывод структуры НС в консоль

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

model.evaluate(x_test, y_test_cat)

n = 5
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(np.argmax(res))

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(y_test[:20])

#
# def predict_digit(img):
#     # изменение рзмера изобржений на 28x28
#     img = img.resize((28, 28))
#     # конвертируем rgb в grayscale
#     img = img.convert('L')
#     img = np.array(img)
#     # изменение размерности для поддержки модели ввода и нормализации
#     img = img.reshape(1, 28, 28, 1)
#     img = img / 255.0
#     # предстказание цифры
#     res = model.predict([img])[0]
#     return np.argmax(res), max(res)
#
#
# class App(tk.Tk):
#     def __init__(self):
#         tk.Tk.__init__(self)
#
#         self.x = self.y = 0
#
#         # Создание элементов
#         self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
#         self.label = tk.Label(self, text="Думаю..", font=("Helvetica", 48))
#         self.classify_btn = tk.Button(self, text="Распознать", command=self.classify_handwriting)
#         self.button_clear = tk.Button(self, text="Очистить", command=self.clear_all)
#
#         # Сетка окна
#         self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
#         self.label.grid(row=0, column=1, pady=2, padx=2)
#         self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
#         self.button_clear.grid(row=1, column=0, pady=2)
#
#         # self.canvas.bind("<Motion>", self.start_pos)
#         self.canvas.bind("<B1-Motion>", self.draw_lines)
#
#     def clear_all(self):
#         self.canvas.delete("all")
#
#     def classify_handwriting(self):
#         HWND = self.canvas.winfo_id()
#         rect = win32gui.GetWindowRect(HWND)  # получаем координату холста
#         im = ImageGrab.grab(rect)
#
#         digit, acc = predict_digit(im)
#         self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')
#
#     def draw_lines(self, event):
#         self.x = event.x
#         self.y = event.y
#         r = 8
#         self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')
#
#
# app = App()
# mainloop()