import matplotlib.pyplot as plt
import numpy as np

# from numpy import mean_squared_error as mse
# from numpy import categorical_crossentropy as cce

# Simple function that calculates mean squared error of two inputs in numpy
def mse(x, y):
    difference = x - y
    sq = np.square(difference)
    return np.mean(sq)

# Simple function that calculates categorical cross entropy of two inputs in numpy
def cce(y_true, y_pred):
    #return -.5 * np.sum(y_true * np.log(y_pred))
    # import pdb; pdb.set_trace()
    y_pred = np.clip(y_pred, 1e-7, None)
    return -np.sum(y_true * np.log(y_pred))

# This is a simple function that plots y = 1 - x
def plot_2d_scatter():
    fs = 15
    #plt.rcParams["figure.figsize"] = (20, 10)
    fig, ax = plt.subplots(figsize=(10, 5))
    # for a simple 2d one hot vector, the label is the first class
    # plot the prediction in the first and second dimension, 

    label = np.array([1, 0])
    x = np.linspace(.0001, 1, 20)
    for x_val in x:
        # predictions will sit on a line, since their sum must be one (to be a valid probability distribution)
        y = 1 - x_val

        rad_cce = cce(label, (x_val, y))
        rad_mse = mse(label, (x_val, y))
        # plot each data point as a circle with radius proportional to the error
        p1 = plt.scatter(x_val, y, s=rad_cce*100, c='r', label='CCE')
        p2 = plt.scatter(x_val, y, s=rad_mse*100, c='b', label='MSE')

    p1 = plt.scatter(-.1, 0, s=25, c='r', label='CCE')
    p2 = plt.scatter(-.1, 0, s=25, c='b', label='MSE')
    plt.xlim(0, 1.1)
    #plt.plot(x, y)
    # add a legend to the plot with red and blue circles
    plt.xlabel('x', fontsize=fs)
    plt.legend((p1, p2), ('CCE', 'MSE'))
    # plot the true label as a gold star
    plt.scatter(x=1, y=0, s=60, c='gold', label='true label', marker='*')
    plt.title('CCE vs MSE errors on a 2d example', fontsize=fs+10)
    plt.show()

# plot_2d_scatter()

# This is a simple function that creates a 3d scatterplot
def plot_3d_scatter():
    from mpl_toolkits.mplot3d import Axes3D

    fs = 15
    #plt.rcParams["figure.figsize"] = (20, 10)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # label is the first class in this one hot vector representing 3 classes   
    label = np.array([1, 0, 0])
    x = np.linspace(.0001, 1, 10)
    y = np.linspace(.0001, 1, 10)

    list_of_points_mse = []
    list_of_points_cce = []
    
    for x_val in x:
        for y_val in y:
            if x_val + y_val < 1: # not a valid point in the output space, since sum > 1
            # predictions will sit on a plane, since their sum must be one (to be a valid probability distribution)
                z = 1 - x_val - y_val
                rad_cce = cce(label, np.array((x_val, y_val, z)))
                rad_mse = mse(label, np.array((x_val, y_val, z)))
                list_of_points_mse.append((x_val, y_val, z, rad_mse))
                list_of_points_cce.append((x_val, y_val, z, rad_cce))

            # plot each data point as a circle with radius proportional to the error
            # p1 = ax.scatter(x_val, y_val, z, s=rad_cce*100, c='r', label='CCE')
            # p2 = ax.scatter(x_val, y_val, z, s=rad_mse*100, c='b', label='MSE')

    # import pdb; pdb.set_trace()
    c1_preds_mse = [i[0] for i in list_of_points_mse]
    c1_preds_cce = [i[0] for i in list_of_points_cce]
    c2_preds_mse = [i[1] for i in list_of_points_mse]
    c2_preds_cce = [i[1] for i in list_of_points_cce]
    c3_preds_mse = [i[2] for i in list_of_points_mse]
    c3_preds_cce = [i[2] for i in list_of_points_cce]
    mse_loss = [i[3]*600 for i in list_of_points_mse]
    cce_loss = [i[3]*450 for i in list_of_points_cce]
    ax.scatter(c1_preds_mse, c2_preds_mse, c3_preds_mse, alpha=.8, s=mse_loss, c='b', label='MSE')
    ax.scatter(c1_preds_cce, c2_preds_cce, c3_preds_cce, alpha=.6, s=cce_loss, c='g', label='CCE')
    p1 = ax.scatter(-1, -1, -1, s=100, c='b', alpha=1, label='MSE')
    p2 = ax.scatter(-1, -1, -1, s=100, c='g', alpha=1, label='CCE')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_zlim(0, 1.1)

    # add a legend to the plot with red and blue circles
    ax.set_xlabel('Class 1 Predicted Probability', fontsize=fs)
    ax.set_ylabel('Class 2 Predicted Probability', fontsize=fs)
    ax.set_zlabel('Class 3 Predicted Probability', fontsize=fs)
    ax.legend((p1, p2), ('CCE', 'MSE'))
    # plot the true label as a gold star
    ax.scatter(1, 0, 0, s=60, c='gold', label='true label', marker='*')
    ax.set_title('CCE vs MSE errors where label is class 1 [1, 0, 0]', fontsize=fs+10)
    fig.tight_layout()
    plt.show()

plot_3d_scatter()
