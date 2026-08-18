
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam


(x_train, y_train_raw), (x_test, y_test_raw) = cifar10.load_data()


class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


y_train = to_categorical(y_train_raw, 10)
y_test = to_categorical(y_test_raw, 10)


fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
axes = axes.ravel()

for i in range(25):
    axes[i].imshow(x_train[i])
    label_index = int(y_train_raw[i][0])
    axes[i].set_title(class_names[label_index], fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.show()


model = Sequential([
    # Layer 1: Conv + MaxPool
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    # Layer 2: Conv + MaxPool
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten & Dense Layers
    Flatten(),
    Dense(units=64, activation='relu'),
    Dense(units=10, activation='softmax')
])


model.summary()


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel compiled successfully and is ready for training!")

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=10,
    validation_data=(x_test, y_test),
    shuffle=True
)


test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

print("\n--- Final Evaluation Metrics ---")
print(f"Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Training vs Validation Accuracy Plot गर्ने
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)

plt.title('CIFAR-10: Training vs Validation Accuracy over Time', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', fontsize=11)
plt.show()


predictions = model.predict(x_test)

predicted_label = np.argmax(predictions[0])
true_label = np.argmax(y_test[0])

print(f"\nमोडलले अनुमान गरेको क्लास: {class_names[predicted_label]}")
print(f"वास्तविक (True) क्लास: {class_names[true_label]}")

# Sample Prediction Display
plt.figure(figsize=(2, 2))
plt.imshow(x_test[0])
plt.title(f"Pred: {class_names[predicted_label]} | True: {class_names[true_label]}")
plt.axis('off')
plt.show()