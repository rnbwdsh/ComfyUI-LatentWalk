import numpy as np
from scipy.stats import norm


def _powspace(start, stop, power, steps):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=steps), power)


def geomspace(start, stop, steps):
    X = np.geomspace(start + 1e-10, stop, steps)
    X[0] = 0
    return X


def reflect_values(X):
    Y = 1 - np.flip(X)
    total_steps = len(X)
    reflect_step = total_steps // 2

    X_a = X[:reflect_step]
    X_b = np.flip(1 - X_a)

    if total_steps % 2 == 0:
        out = np.concatenate([X_a, X_b])
    else:
        mid = total_steps // 2
        mean_val = np.mean([X[mid], Y[mid]]).reshape(1)
        out = np.concatenate([X_a, mean_val, X_b])
    return np.round(out, 5)


def circle_points(steps):
    # Angle in radians from start to stop
    # Linspace to get 'steps' number of points between start_angle and stop_angle
    # x and y coordinates for the points on the circle
    x = (np.cos(np.linspace(np.radians(180), np.radians(0), steps)) + 1) / 2
    # y = np.sin(theta)

    return x


def hinge_points(start, stop, steps, hinge):
    if steps % 2 == 0:
        A = np.linspace(start, hinge, num=steps // 2)
        B = np.linspace(hinge, stop, num=(steps // 2 + 1))
        out = np.concatenate([A, B[1:]])  # remove duplicated end point
    else:
        A = np.linspace(start, hinge, num=steps // 2)
        B = np.linspace(hinge, stop, num=(steps // 2) + 2)
        out = np.concatenate([A, B[1:]])  # remove duplicated end point

    return out


def normspace(start, stop, steps, factor):
    X = np.linspace(start, stop, int(steps - 2))
    Y = norm.cdf(X, loc=0.5, scale=factor)
    # Insert 0 and 1 strengths so that starting and end images are unchanged
    Y = np.insert(Y, 0, 0)
    Y = np.append(Y, 1)

    return Y


TRAVEL_MODES = {
    'linear': lambda steps, ___: np.linspace(0, 1, steps),
    'circle': lambda steps, ___: (np.cos(np.linspace(np.radians(180), np.radians(0), steps)) + 1) / 2,
    'quadratic': lambda steps, __: _powspace(0.0, 1.0, 2, steps),
    'cubic': lambda steps, _: _powspace(0.0, 1.0, 3, steps),
    'quartic': lambda steps, _: _powspace(0.0, 1.0, 4, steps),
    'geometric': lambda steps, _: geomspace(0.0, 1.0, steps),
    'hinge': lambda steps, factor: hinge_points(0.0, 1.0, steps, factor),
    'norm': lambda steps, factor: normspace(0.0, 1.0, steps, factor),
}
