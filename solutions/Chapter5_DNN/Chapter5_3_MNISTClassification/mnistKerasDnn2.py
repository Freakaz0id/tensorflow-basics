from typing import Tuple

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical


def get_dataset(
    num_features: int, num_classes: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, num_features).astype(np.float32)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_classes: int) -> Sequential:
    init_w = TruncatedNormal(mean=0.0, stddev=0.01)
    init_b = Constant(value=0.0)

    model = Sequential()
    model.add(
        Dense(
            units=500,
            kernel_initializer=init_w,
            bias_initializer=init_b,
            input_shape=(num_features,),
        )
    )
    model.add(Activation("relu"))
    model.add(Dense(units=250, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(
        Dense(
            units=num_classes,
            kernel_initializer=init_w,
            bias_initializer=init_b,
        )
    )
    model.add(Activation("softmax"))
    model.summary()

    return model


def main() -> None:
    num_features = 784
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = get_dataset(num_features, num_classes)

    model = build_model(num_features, num_classes)

    # opt = Adam(learning_rate=0.002)  # Scores on test set: [0.12499380111694336, 0.9779999852180481]
    # opt = SGD(learning_rate=0.002) # Scores on test set: [0.11796435713768005, 0.9661999940872192]
    # opt = RMSprop(learning_rate=0.002)  # Scores on test set: [0.29667389392852783, 0.9674000144004822]

    # opt = Adam(learning_rate=0.001)  # Scores on test set: [0.09296467155218124, 0.9815000295639038]
    # opt = SGD(learning_rate=0.001) # Scores on test set: [0.20199905335903168, 0.9409000277519226]
    opt = RMSprop(
        learning_rate=0.001
    )  # Scores on test set: [0.19755497574806213, 0.982200026512146]

    # opt = Adam(learning_rate=0.0009)  # Scores on test set: [0.10250154137611389, 0.9789000153541565]
    # opt = SGD(learning_rate=0.0009)  # Scores on test set: [0.22731703519821167, 0.9347000122070312]
    # opt = RMSprop(learning_rate=0.0009)  # Scores on test set: [0.17982593178749084, 0.9797000288963318]

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.fit(
        x=x_train,
        y=y_train,
        epochs=20,
        batch_size=256,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    scores = model.evaluate(x=x_test, y=y_test, verbose=0)
    print(f"Scores on test set: {scores}")


if __name__ == "__main__":
    main()
