import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Wczytanie danych
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data = train_datagen.flow_from_directory('C:/Users/piotr/OneDrive/Pulpit/Python kurs/data', target_size=(64,64), batch_size=32, class_mode='categorical')
val_datagen = ImageDataGenerator(rescale=1./255)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('C:\\Users\\piotr\\OneDrive\\Pulpit\\Python kurs\\data', target_size=(64,64), batch_size=32, class_mode='categorical')
val_data = val_datagen.flow_from_directory('C:\\Users\\piotr\\OneDrive\\Pulpit\\Python kurs\\data', target_size=(64,64), batch_size=32, class_mode='categorical')

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=val_data)

test_datagen = ImageDataGenerator(rescale=1./255)

# Stworzenie modelu
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(train_data, epochs=20, validation_data=val_data)

# Zapisanie modelu
model.save('model.h5')

# Wykres uczenia
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')


# Testowanie modelu
test_loss, test_acc = model.evaluate(test_generator)




print('Test accuracy:', test_acc)

# Predykcja na 10 losowych obrazach
x, y_true = test_data.next()
y_pred = model.predict(x)
y_pred_labels = np.argmax(y_pred, axis=1)
class_names = ['0', 'A', 'B', 'C']
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12,6), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(x[i])
    true_label = class_names[np.argmax(y_true[i])]
    pred_label = class_names[y_pred_labels[i]]
    ax.set_title(f"True label: {true_label}, Pred label: {pred_label}")
plt.show()
# Zapisanie modelu do pliku
model.save('asl_model.h5')
import matplotlib.pyplot as plt

# Wykres krzywej uczenia
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Krzywa uczenia')
plt.xlabel('Epoki')
plt.ylabel('Wartość')
plt.legend()
plt.show()
# Ocena jakości modelu na danych testowych
test_loss, test_acc = model.evaluate(test_ds)

print(f'Dokładność (accuracy) na danych testowych: {test_acc:.2%}')
# Wyświetlenie 10 losowych obrazów z zestawu testowego wraz z przewidzianymi etykietami
import numpy as np

# Pobranie losowych obrazów z zestawu testowego
images, labels = [], []
for image, label in test_ds.unbatch().take(10):
    images.append(image.numpy())
    labels.append(label.numpy())

images = np.array(images)
labels = np.array(labels)

# Przewidywanie etykiet dla losowych obrazów
predicted_labels = model.predict(images).argmax(axis=1)

# Wyświetlenie obrazów wraz z etykietami
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()
for i in range(10):
    img = images[i]
    true_label = labels[i]
    pred_label = predicted_labels[i]
    axs[i].imshow(img)
    axs[i].set_title(f"True label: {true_label}, Pred label: {pred_label}")
plt.show()
