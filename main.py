import math
from typing import Callable, Tuple

import numpy as np
import pandas as pd


def normalize(lst: np.ndarray):
    mean = lst.mean()
    std = lst.std()
    if std != 0:
        lst = (lst - mean) / std
    else:
        lst = np.zeros(len(lst))
    return lst, mean, std


def cost_function(
    func: Callable[[np.ndarray], float], xs: np.ndarray, ys: np.ndarray
) -> float:
    assert xs.shape[0] == ys.shape[0], "Mismatched number of samples"
    assert xs.ndim == 2, "xs must be a 2D array (samples × features)"
    assert ys.ndim == 1, "ys must be a 1D array"

    predictions = np.array([func(x) for x in xs])
    errors = predictions - ys
    cost = np.sum(errors**2) / (2 * len(xs))
    return cost


def compute_gradient(
    xs: np.ndarray, ys: np.ndarray, w: np.ndarray, b: float
) -> Tuple[np.ndarray, float]:
    assert xs.shape[0] == ys.shape[0], "Mismatched number of samples"
    assert xs.ndim == 2, "xs must be a 2D array (samples × features)"
    assert ys.ndim == 1, "ys must be a 1D array"

    m, n = xs.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        xi = xs[i]
        yi = ys[i]
        prediction = np.dot(w, xi) + b
        error = prediction - yi
        dj_dw += error * xi
        dj_db += error

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(
    inputs: np.ndarray,
    output: np.ndarray,
    w_in: np.ndarray,
    b_in: float,
    learning_step: float,
    num_iterations: int,
    compute_gradient: Callable[
        [np.ndarray, np.ndarray, np.ndarray, float], Tuple[np.ndarray, float]
    ],
    cost_function: Callable[
        [Callable[[np.ndarray], float], np.ndarray, np.ndarray], float
    ],
) -> Tuple[np.ndarray, float]:
    """
    Args:
      w_in,b_in: initial values of model parameters
    """
    w, b = w_in, b_in
    for i in range(num_iterations):
        dj_dw, dj_db = compute_gradient(inputs, output, w, b)

        w -= learning_step * dj_dw
        b -= learning_step * dj_db

        if i % 10 == 0:
            print(
                f"Iteration = {i}, w = {w}, b = {b}, cost = {cost_function(lambda x: np.dot(x, w) + b, inputs, output)}"
            )

    return w, b


def main():
    df = pd.read_csv("./resources/housing.csv")
    inputs_columns = ["total_rooms", "total_bedrooms"]
    output_column = "median_house_value"

    df = df.dropna(subset=inputs_columns + [output_column])

    inputs: np.ndarray = df[inputs_columns].to_numpy()
    output: np.ndarray = df[output_column].to_numpy()

    print(inputs)

    for i, _ in enumerate(inputs_columns):
        inputs[:, i], *_ = normalize(inputs[:, i])

    output, mu, std = normalize(output)

    w, b = gradient_descent(
        inputs,
        output,
        np.array([1.0, 2.0]),
        2,
        1e-2,
        1000,
        compute_gradient,
        cost_function,
    )
    print(w, b)
    func = lambda x: np.dot(w, x) + b
    print(cost_function(func, inputs, output))


if __name__ == "__main__":
    main()
