from typing import Tuple

import numpy as np
from tensorflow.keras.datasets import boston_housing


def get_dataset() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    x_train = x_train.astype(np.float32)
    y_train = y_train.reshape((-1, 1)).astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.reshape((-1, 1)).astype(np.float32)
    return (x_train, y_train), (x_test, y_test)


def main() -> None:
    (x_train, y_train), (x_test, y_test) = get_dataset()

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")


if __name__ == "__main__":
    main()
