import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import DenseNet201, ResNet101, Xception

# Параметры модели
input_shape = (128, 128, 3)  # Размеры изображений должны соответствовать входу DenseNet201
num_classes = 10  # Количество классов (людей)

# Загрузка предобученной модели DenseNet201 без верхних слоев
base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

# Замораживаем базовые слои
base_model.trainable = False

# Создание модели
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обзор модели
model.summary()

# Генераторы данных для обучения и тестирования
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=35,          # Повороты изображений на 10 градусов
    width_shift_range=0.5,      # Сдвиг изображений по ширине на 10%
    height_shift_range=0.5,     # Сдвиг изображений по высоте на 10%
    shear_range=0.1,            # Сдвиг (shear) на 10%
    zoom_range=0.40,            # Увеличение/уменьшение изображения на 10%
    horizontal_flip=False,      # Отключаем горизонтальное отражение (не актуально для подписей)
    brightness_range=[0.8, 1.2],  # Варьируем яркость изображений
    channel_shift_range=50.0,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'train',  # Путь к директории с тренировочными данными
    target_size=(128, 128),  # Размер изображений должен соответствовать input_shape
    batch_size=5,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'test',  # Путь к директории с валидационными данными
    target_size=(128, 128),
    batch_size=5,
    class_mode='categorical'
)

# Коллбэк для сохранения модели с наибольшей точностью
checkpoint = ModelCheckpoint(
    'best_model_import.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Обучение модели с коллбэками
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=250,
    callbacks=[checkpoint]
)

# Сохранение окончательной модели
model.save('model_import.keras')
