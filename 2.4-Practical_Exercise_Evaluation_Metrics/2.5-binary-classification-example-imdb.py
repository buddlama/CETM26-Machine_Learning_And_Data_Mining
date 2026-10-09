"""
Classifying movie reviews: a binary classification example (IMDB dataset)

Full, runnable version of the code walked through in the learning journal entry.
Requires: keras, numpy, matplotlib  (pip install keras numpy matplotlib tensorflow)
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.datasets import imdb


# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
# num_words=10000 keeps only the 10,000 most frequent words; rarer words are
# discarded so we work with vector data of manageable size.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("First training review (word indices):", train_data[0])
print("First training label:", train_labels[0])
print("Max word index used:", max(max(sequence) for sequence in train_data))


# ---------------------------------------------------------------------------
# 2. (Optional) Decode a review back to readable English
# ---------------------------------------------------------------------------
word_index = imdb.get_word_index()
reverse_word_index = dict((value, key) for (key, value) in word_index.items())
# Indices are offset by 3 because 0, 1, 2 are reserved for "padding",
# "start of sequence", and "unknown".
decoded_review = " ".join(
    reverse_word_index.get(i - 3, "?") for i in train_data[0]
)
print("Decoded first review:", decoded_review[:100])


# ---------------------------------------------------------------------------
# 3. Prepare the data: multi-hot encode the integer sequences
# ---------------------------------------------------------------------------
def multi_hot_encode(sequences, num_classes):
    # Creates an all-zero matrix of shape (len(sequences), num_classes)
    results = np.zeros((len(sequences), num_classes))
    for i, sequence in enumerate(sequences):
        # Sets specific indices of results[i] to 1s
        results[i][sequence] = 1.0
    return results


x_train = multi_hot_encode(train_data, num_classes=10000)
x_test = multi_hot_encode(test_data, num_classes=10000)

print("Example multi-hot vector (first 20 values):", x_train[0][:20])

# Labels are already 0s and 1s, just convert to float32
y_train = train_labels.astype("float32")
y_test = test_labels.astype("float32")


# ---------------------------------------------------------------------------
# 4. Build the model
# ---------------------------------------------------------------------------
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# ---------------------------------------------------------------------------
# 5. Set aside a validation set
# ---------------------------------------------------------------------------
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# ---------------------------------------------------------------------------
# 6. Train for 20 epochs while monitoring the validation set
# ---------------------------------------------------------------------------
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)

history_dict = history.history
print("Tracked metrics:", history_dict.keys())


# ---------------------------------------------------------------------------
# 7. Plot training vs validation loss
# ---------------------------------------------------------------------------
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs_range = range(1, len(loss_values) + 1)

plt.figure()
plt.plot(epochs_range, loss_values, "r--", label="Training loss")
plt.plot(epochs_range, val_loss_values, "b", label="Validation loss")
plt.title("[IMDB] Training and validation loss")
plt.xlabel("Epochs")
plt.xticks(list(epochs_range))
plt.ylabel("Loss")
plt.legend()
plt.savefig("imdb_loss.png")
plt.show()


# ---------------------------------------------------------------------------
# 8. Plot training vs validation accuracy
# ---------------------------------------------------------------------------
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]

plt.plot(epochs_range, acc, "r--", label="Training acc")
plt.plot(epochs_range, val_acc, "b", label="Validation acc")
plt.title("[IMDB] Training and validation accuracy")
plt.xlabel("Epochs")
plt.xticks(list(epochs_range))
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("imdb_accuracy.png")
plt.show()

# Looking at the two plots: validation loss/accuracy typically peak around
# epoch 4, after which the model starts overfitting (training metrics keep
# improving, validation metrics get worse). So we retrain from scratch below
# using only 4 epochs.


# ---------------------------------------------------------------------------
# 9. Retrain a fresh model for 4 epochs on the FULL training set
# ---------------------------------------------------------------------------
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(x_train, y_train, epochs=4, batch_size=512)

results = model.evaluate(x_test, y_test)
print("Test loss, test accuracy:", results)


# ---------------------------------------------------------------------------
# 10. Use the trained model to predict on new data
# ---------------------------------------------------------------------------
predictions = model.predict(x_test)
print("First 10 predicted probabilities (positive sentiment):")
print(predictions[:10])