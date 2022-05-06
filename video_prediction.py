# import numpy as np
# import matplotlib.pyplot as plt


# def LoadData(dataType):
#     #directory = 'dataset_small'
#     directory = 'dataset'
#     frames = np.load(directory+'/frames.npy')[..., 0:3]
#     counts = np.load(directory+'/count.npy')

#     train_size, valid_size = int(
#         frames.shape[0] * 0.6), int(frames.shape[0] * 0.2)

#     if dataType == "train":
#         train_x = frames[: train_size]
#         train_y = counts[: train_size]

#         valid_x = frames[train_size: train_size + valid_size]
#         valid_y = counts[train_size: train_size + valid_size]
#         train_x = np.array(train_x)
#         train_y = np.array(train_y)
#         valid_x = np.array(valid_x)
#         valid_y = np.array(valid_y)
#         print(train_x.shape)
#         print(train_y.shape)
#         print(valid_x.shape)
#         print(valid_y.shape)
#         return train_x, train_y, valid_x, valid_y

#     else:
#         test_x = frames[train_size + valid_size:]
#         test_y = counts[train_size + valid_size:]
#         test_x = np.array(test_x)
#         test_y = np.array(test_y)
#         print(test_x.shape)
#         print(test_y.shape)

#         return test_x, test_y


# def DrawGraph(train, valid):
#     # removing first element because usually that is a very big number
#     del train[0]
#     del valid[0]

#     epochs = list(range(1, len(train)+1))

#     plt.plot(epochs, train, label="Train")

#     plt.plot(epochs, valid, label="Valid")

#     # naming the x axis
#     plt.xlabel('Epochs')
#     # naming the y axis
#     plt.ylabel('Loss')
#     # giving a title to my graph
#     plt.title('Train and validation loss')

#     # show a legend on the plot
#     plt.legend()

#     # function to show the plot
#     # plt.show()
#     # plt.savefig('figures/loss-{}.png'.format(exp_name))
#     plt.savefig('results/loss.png')
#     plt.clf()


import random
num_videos = 20
filepath = 'predictions/depth.txt'

with open(filepath) as f:
    lines = f.readlines()

loss = 0
n = len(lines)
for i in range(num_videos):
    first = random.randint(0, n)
    second = random.randint(0, n)

    line1 = lines[first].strip().split('             ')
    line2 = lines[second].strip().split('             ')

    real = abs(float(line1[0]) - float(line2[0]))
    prediction = abs(float(line1[1]) - float(line2[1]))

    loss += (real-prediction)**2

print("mse loss for entire video with depth: ", loss/num_videos)
