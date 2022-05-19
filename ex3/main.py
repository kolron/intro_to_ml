import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    filename = r'D:\school\machine learning\ex3\two_circle.csv'
    data = pd.read_csv(filename,header=None)
    print(data)

    # make the dataset linearly separable
    data = np.asmatrix(data, dtype='float64')
    return data

def plotCircle(X,y):
    xs = []
    ys = []
    clas = []
    for point2 in zip(X, y):
        xs.append(point2[0][0])
        ys.append(point2[0][1])
        clas += ['r' if point2[1] == 1 else 'b']
    plt.scatter(xs, ys, c=clas)

def perceptron(data, num_iter):
    # separating features (x,y) and labeling (1,-1)
    features = data[:, :-1]
    labels = data[:, -1]

    # weights equal zero to start with
    w = np.zeros(shape=(1, features.shape[1] + 1))

    misclassified_ = [] # how many misclassified points

    for epoch in range(num_iter):
        misclassified = 0
        for x, label in zip(features, labels):
            x = np.insert(x, 0, 1)
            y = np.dot(w, x.transpose())
            target = 1.0 if (y > 0) else -1.0

            delta = (label.item(0, 0) - target)

            if (delta):  # misclassified
                misclassified += 1
                w += (delta * x)

        misclassified_.append(misclassified)
    return (w, misclassified_)


if __name__ == '__main__':
    num_iter = 100
    data = load_data()


    w, misclassified_ = perceptron(data, num_iter)
    print(f'Final weights vector: {w[0][1:]}')
    epochs = np.arange(1, num_iter+1)
    plt.plot(epochs, misclassified_)
    plt.xlabel('iterations')
    plt.ylabel('misclassified')
    plt.show()
