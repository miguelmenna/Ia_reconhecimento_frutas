import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Definir o diretório base do dataset
base_dir = "/content/drive/MyDrive/frutas"  # Substitua pelo caminho real do seu dataset

# Verifique se o diretório existe
if not os.path.exists(base_dir):
    raise ValueError(f"O diretório {base_dir} não existe. Verifique o caminho.")

# Configurar os Geradores de Dados com mais augmentations
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Separar uma parte para validação
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Mapear classes
class_names = list(train_generator.class_indices.keys())
print(f"Classes disponíveis: {class_names}")

# Construção do Modelo CNN com camadas adicionais
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),  # Dropout para evitar overfitting
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Regularização L2
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# Compilação do Modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks: EarlyStopping e ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Treinamento do Modelo
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Avaliação do Modelo
validation_loss, validation_acc = model.evaluate(validation_generator)
print(f"Validation loss: {validation_loss}")
print(f"Validation accuracy: {validation_acc}")

# Visualização da Acurácia e Perda durante o Treinamento
plt.figure(figsize=(12, 6))

# Gráfico de Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia de treino')
plt.plot(history.history['val_accuracy'], label='Acurácia de validação')
plt.title('Acurácia durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Gráfico de Perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda de treino')
plt.plot(history.history['val_loss'], label='Perda de validação')
plt.title('Perda durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.show()

# Prever para o conjunto de validação
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Geração do relatório de classificação
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred, labels=np.arange(len(class_names)), target_names=class_names))

# Matriz de Confusão
cm = confusion_matrix(validation_generator.classes, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Verificar se o caminho da imagem está correto
img_path = '/content/drive/MyDrive/frutasTeste/fruta.jpg'  # Substitua pelo caminho de uma imagem para predição

if os.path.exists(img_path):
    # Carregar a imagem
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Fazer a predição
    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)
    print(f'Classe prevista: {class_names[pred_class[0]]}')
else:
    print(f"O arquivo {img_path} não foi encontrado. Verifique o caminho.")

# Salvamento do Modelo Treinado
model.save('fruits_model.h5')
print("Modelo salvo com sucesso!")
