import numpy as np


def LoadData(dataType):
    directory = 'dataset_small'
    frames = np.load(directory+'/frames.npy')[..., 0:3]
    counts = np.load(directory+'/count.npy')

    train_size, valid_size = int(
        frames.shape[0] * 0.6), int(frames.shape[0] * 0.2)

    if dataType == "train":
        train_x = frames[: train_size]
        train_y = counts[: train_size]

        valid_x = frames[train_size: train_size + valid_size]
        valid_y = counts[train_size: train_size + valid_size]
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        valid_x = np.array(valid_x)
        valid_y = np.array(valid_y)
        return train_x, train_y, valid_x, valid_y

    else:
        test_x = frames[train_size + valid_size:]
        test_y = counts[train_size + valid_size:]
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        return test_x, test_y
