import pandas as pd
from functools import cmp_to_key
import matplotlib.pyplot as plt
import numpy as np

line_error = {}

# def abline(slope, intercept):
#     """Plot a line from slope and intercept"""
#     axes = plt.gca()
#     x_vals = np.array(axes.get_xlim())
#     y_vals = intercept + slope * x_vals
#     plt.plot(x_vals, y_vals, '--')
#     plt.show()


def compare(x, y):
    val_x = line_error[x]
    val_y = line_error[y]
    return val_x - val_y


def load_data():
    filename = r'D:\school\machine learning\ex3\four_circle.csv'
    data = pd.read_csv(filename, header=None)
    print(data)
    X = []
    y = []
    # make the dataset linearly separable
    data = np.asmatrix(data, dtype='float64')
    data_arr = np.squeeze(np.asarray(data))
    for i in range(len(data_arr)):
        X.append((data_arr[i][0], data_arr[i][1]))
        y.append(data_arr[i][2])
    return X, y, data


##
def getLines(data):
    lines = {}
    for line in data:
        line = np.squeeze(np.asarray(line))
        x = line[0]
        y = line[1]
        for other in data:
            other = np.squeeze(np.asarray(other))
            other_x = other[0]
            other_y = other[1]
            slope = (y - other_y) / (x - other_x) if x != other_x else 0
            # y = mx + b
            # b = y - mx
            b = y - (slope * x)
            line_eq = (slope, b)
            lines.setdefault(line_eq, '1')
    final_lines = [key for key in lines.keys()]
    return final_lines


def bestLines(lines, X, y):
    misclassification = {}
    for line in lines:
        if line not in misclassification.keys():
            misclassification[line] = 0
        for point in zip(X, y):
            point_x = point[0][0]
            point_y = point[0][1]
            # y     =        mx       +   b
            line_y = (line[0] * point_x) + line[1]
            if point_y > line_y:
                classification = 1
            else:
                classification = -1
            if classification != point[1]:  # no good
                misclassification[line] += 1

    return misclassification


def plotCircle(X, y):
    xs = []
    ys = []
    clas = []
    for point2 in zip(X, y):
        xs.append(point2[0][0])
        ys.append(point2[0][1])
        clas += ['r' if point2[1] == 1 else 'b']
    plt.scatter(xs, ys, c=clas)


if __name__ == '__main__':
    X, y, data = load_data()
    # plotCircle(X,y)
    # plt.show()
    lines = getLines(data)

    line_error = bestLines(lines, X, y)

    # now we have the best lines sorted, least error on each point first.
    best_lines = sorted(lines, key=cmp_to_key(compare))
    best_8_lines = best_lines[0:8]
    print(best_8_lines)
