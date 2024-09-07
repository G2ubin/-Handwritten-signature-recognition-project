import sys
import os
import time
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

class MainWindow(QWidget):
    
    def __init__(self):
        super().__init__()
        self.names = ["Алина", "Андрей", "Юлия", "Матвей", "Настя", "Ольга", "Платон", "Полина", "Семён", "Владимир"]
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose model", os.path.expanduser('~'), "Model files (*.keras *.h5 )")
        if not file_path or not os.path.exists(file_path):
            self.warningText = "Модель не выбрана или не найдена"
            print(self.warningText)
            sys.exit()
        
        self.model = load_model(file_path)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Распознавание подписи')
        self.setGeometry(100, 100, 800, 600)

        self.button = QPushButton('Выбрать изображение для распознавания', self)
        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(QPixmap("empty.png").scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
        self.timeLabel = QLabel("")
        self.percentLabel = QLabel("")
        self.outputLabel = QLabel("")

        layout = QVBoxLayout()
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.button)
        layout.addWidget(self.timeLabel)
        layout.addWidget(self.percentLabel)
        layout.addWidget(self.outputLabel)
        self.setLayout(layout)

        self.button.clicked.connect(self.on_button_click)

    def on_button_click(self):
        imagePath, _ = QFileDialog.getOpenFileName(self, "Open Image", os.path.expanduser('~'), "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not imagePath:
            return

        # Отображаем изображение
        pixmap = QPixmap(imagePath)
        self.imageLabel.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))

        # Предсказание
        startTime = time.time()
        prediction = self.predict_image(imagePath)
        endTime = time.time()

        # Вывод результатов
        self.timeLabel.setText(f"Время распознавания: {endTime - startTime:.3f} сек.")
        
        max_index = np.argmax(prediction[0])
        predicted_name = self.names[max_index]
        confidence = prediction[0][max_index] * 100
        self.percentLabel.setText(f"Вероятность росписи {predicted_name}: {confidence:.3f}%")

        # Вывод вероятностей для всех классов
        output_text = "\n".join([f"{self.names[i]}: {prediction[0][i]*100:.3f}%" for i in range(len(self.names))])
        self.outputLabel.setText(output_text)

    def predict_image(self, image_path):
        image = load_img(image_path, target_size=(128, 128))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return self.model.predict(image_array)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
"""
СТАРАЯ ВЕРСИЯ

import sys
import os
import time
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image, ExifTags
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np


class MainWindow(QWidget):
    
    def __init__(self):
        super().__init__()
        self.names = ["Алина", "Андрей", "Юлия", "Матвей", "Настя", "Ольга", "Платон", "Полина", "Семён", "Владимир"]
        self.warningText = ""
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose model", os.path.expanduser('~'), "Model files (*.keras *.h5 )")
        if file_path:
            self.model_path = file_path
        else:
            self.warningText ="Модель не выбрана"
            return
        
        print(self.model_path)
        self.model = self.load_model_once(self.model_path)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Распознавание подписи')
        self.setGeometry(100, 100, 800, 600)

        self.button = QPushButton('Выбрать изображение для распознавания', self)
        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(QPixmap("empty.png").scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
        self.timeLabel = QLabel("")
        self.percentLabel = QLabel("")
        self.outputLabel = QLabel("")
        self.warningLabel = QLabel(self.warningText)

        layout = QVBoxLayout()
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.button)
        layout.addWidget(self.timeLabel)
        layout.addWidget(self.percentLabel)
        layout.addWidget(self.outputLabel)
        layout.addWidget(self.warningLabel)
        self.setLayout(layout)

        self.button.clicked.connect(self.on_button_click)

    def load_model_once(self, model_path):
        if os.path.exists(model_path):
            return load_model(model_path)
        else:
            self.warningText = "Модель не найдена."
            sys.exit()

    def on_button_click(self):
        imagePath, _ = QFileDialog.getOpenFileName(self, "Open Image", os.path.expanduser('~'), "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if imagePath:
            startTime = time.time()

            pixmap = QPixmap(imagePath)
            self.imageLabel.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))

            prediction = self.predict_image(imagePath)
            print(f"Алина: {prediction[0][0]*100:.3f}%\n\
Андрей: {prediction[0][1]*100:.3f}%\n\
Юлия: {prediction[0][2]*100:.3f}%\n\
Матвей: {prediction[0][3]*100:.3f}%\n\
Настя: {prediction[0][4]*100:.3f}%\n\
Ольга: {prediction[0][5]*100:.3f}%\n\
Платон: {prediction[0][6]*100:.3f}%\n\
Полина: {prediction[0][7]*100:.3f}%\n\
Семён: {prediction[0][8]*100:.3f}%\n\
Владимир: {prediction[0][9]*100:.3f}%\n")

            endTime = time.time()
            
            max_index = np.argmax(prediction[0])
            self.timeLabel.setText(f"Время распознавания: {endTime - startTime:.3f} сек.")
            if 0 <= max_index < len(self.names) and 0 <= max_index < len(prediction[0]):
                self.percentLabel.setText(f"Вероятность росписи {self.names[max_index]} на фото: {(prediction[0][max_index]*100):.3f}%")
            else:
                print("Ошибка: max_index вне диапазона")
        
    def predict_image(self, image_path):
        image = load_img(image_path, target_size=(128, 128))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = self.model.predict(image_array)
        return prediction

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
"""