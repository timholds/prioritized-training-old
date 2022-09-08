from tensorflow import keras
from tensorflow.keras import layers

class ConvModel(keras.Model):
  
  def create_model(self, num_classes=10): 

    input_shape = (28, 28, 1)
    model = keras.Sequential(
      [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")]
    )

    return model

