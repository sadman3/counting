import numpy as np
import matplotlib.pyplot as plt
import random


def LoadData(dataType):
    # #directory = 'dataset_small'
    directory = 'dataset'
    # frames = np.load(directory+'/frames.npy')[..., 0:3]
    # #frames /= 255.0
    # counts = np.load(directory+'/count.npy')

    # train_size, valid_size = int(
    #     frames.shape[0] * 0.6), int(frames.shape[0] * 0.2)

    if dataType == "train":
        train_x = np.load(directory + '/train/frames.npy')[..., 0:3]
        train_y = np.load(directory + '/train/count.npy')

        indices = list(range(0, train_x.shape[0]-1))
        random.shuffle(indices)

        shuffled_x = []
        shuffled_y = []

        for i in indices:
            shuffled_x.append(train_x[i])
            shuffled_y.append(train_y[i])

        train_x = np.array(shuffled_x)
        train_y = np.array(shuffled_y)

        valid_x = np.load(directory + '/valid/frames.npy')[..., 0:3]
        valid_y = np.load(directory + '/valid/count.npy')

        # np.save(directory + '/train/frames.npy', train_x)
        # np.save(directory + '/train/count.npy', train_y)
        # np.save(directory + '/valid/frames.npy', valid_x)
        # np.save(directory + '/valid/count.npy', valid_y)
        print(train_x.shape)
        print(train_y.shape)
        print(valid_x.shape)
        print(valid_y.shape)
        return train_x, train_y, valid_x, valid_y

    else:
        test_x = np.load(directory + '/test/frames.npy')[..., 0:3]
        test_y = np.load(directory + '/test/count.npy')

        # np.save(directory + '/test/frames.npy', test_x)
        # np.save(directory + '/test/count.npy', test_y)

        return test_x, test_y


def DrawGraph(train, valid):
    # removing first element because usually that is a very big number
    del train[0]
    del valid[0]

    epochs = list(range(1, len(train)+1))

    plt.plot(epochs, train, label="Train")

    plt.plot(epochs, valid, label="Valid")

    # naming the x axis
    plt.xlabel('Epochs')
    # naming the y axis
    plt.ylabel('Loss')
    # giving a title to my graph
    plt.title('Train and validation loss')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    # plt.show()
    # plt.savefig('figures/loss-{}.png'.format(exp_name))
    plt.savefig('results/loss.png')
    plt.clf()
