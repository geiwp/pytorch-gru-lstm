import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Sequential, Model, layers, metrics, optimizers, losses
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, GRU, Dense
# from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

# 加载IMDB数据集
max_features = 10000
maxlen = 100
batch_size = 32

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# 构建GRU模型
model = Sequential()
model.add(keras.layers.Embedding(max_features, 128))
model.add(keras.layers.GRU(128, dropout=0.2, recurrent_dropout=0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_data=(x_test, y_test))

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 计算并打印准确度
y_pred = model.predict(x_test)
y_pred_binary = (y_pred > 0.5).astype(int)
overall_accuracy = np.mean(np.equal(y_test, y_pred_binary))
print("Overall Accuracy:", overall_accuracy)

class_accuracies = []
for class_label in range(2):
    class_indices = np.where(y_test == class_label)
    class_accuracy = np.mean(np.equal(y_test[class_indices], y_pred_binary[class_indices]))
    class_accuracies.append(class_accuracy)
    print(f"Class {class_label} Accuracy:", class_accuracy)

plt.subplot(1, 2, 2)
plt.bar(['Class 0', 'Class 1'], class_accuracies)
plt.ylabel('Accuracy')
plt.title('Class-wise Accuracy')

plt.show()
