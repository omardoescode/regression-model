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
    w = np.zeros(inputs.shape[1])
    b = 0
    for i in range(num_iterations):
        dj_dw, dj_db = compute_gradient(inputs, output, w, b)

        w -= learning_step * dj_dw
        b -= learning_step * dj_db

        if i % 300 == 0:
            print(
                f"Iteration = {i}, w = {w}, b = {b}, cost = {cost_function(lambda x: np.dot(x, w) + b, inputs, output)}"
            )

    return w, b


def predict(
    func: Callable[[np.ndarray], float], input_stats, output_stat, input: np.ndarray
):
    normalized_input = np.array(
        [
            (input[i] - input_stats[i]["mean"]) / input_stats[i]["std"]
            for i in range(len(input))
        ]
    )
    normalized = func(normalized_input)
    return output_stat["mean"] + normalized * output_stat["std"]


def main():
    df = pd.read_csv("./resources/social-media-addiction.csv")
    inputs_columns = [
        "Avg_Daily_Usage_Hours",
        "Age",
        "Mental_Health_Score",
        "Sleep_Hours_Per_Night",
    ]
    output_column = "Addicted_Score"

    df = df.dropna(subset=inputs_columns + [output_column])

    # Outlier removal using z-score
    for col in inputs_columns + [output_column]:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(np.abs(df[col] - mean) / std) <= 2]  # Keep values with |z| <= 2

    # Convert to NumPy arrays
    inputs = df[inputs_columns].to_numpy()
    output = df[output_column].to_numpy()

    input_stats = []
    for i, col in enumerate(inputs_columns):
        inputs[:, i], mu, std = normalize(inputs[:, i])
        input_stats.append({"mean": mu, "std": std})

    output_stats = {}
    output, output_stats["mean"], output_stats["std"] = normalize(output)

    w, b = gradient_descent(
        inputs,
        output,
        1e-2,
        10000,
        compute_gradient,
        cost_function,
    )
    func = lambda x: np.dot(w, x) + b
    print(cost_function(func, inputs, output))

    print("Testing some values: ")
    print(predict(func, input_stats, output_stats, np.array([8, 14, 6, 8])))


if __name__ == "__main__":
    main()
