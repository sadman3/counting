import os
import numpy as np
import matplotlib.pyplot as plt


def dataset_statistics():
    directory = "/media/rpal/Drive_10TB1/sadman/pouring_data/"

    file_list = [os.path.join(directory, item)
                 for item in os.listdir(directory) if 'npy' in item]

    print("---- Dataset Details ----\n")
    num_videos = len(file_list)
    print("number of videos: ", num_videos)

    total_frame = 0
    video_arr = []

    massOfBlock = 14.375e-03
    single_block_weight = massOfBlock * 9.8 / 4.448
    i = 0
    for file in file_list:
        data = np.load(file)
        total_frame += data.shape[0]

        if i > 175 and i < 185 and data.shape[0] < 700:
            # temp = np.round((-1 * data[:, 1])/single_block_weight)
            # video_arr.append(temp)
            video_arr.append(data[:, 6])
        i += 1
        # arr = []
        # for i in range(data.shape[0]):
        #     receiver_weight = data[i][1]
        #     arr.append(receiver_weight)
        # video_arr.append(arr)

    print("total number of frame in the dataset: ", total_frame)
    print("train, test and valid ratio = 60:20:20")
    print("#frame used in training: ", int(total_frame*0.6))
    print("#frame used in training: ", int(total_frame*0.2))
    print("#frame used in training: ", total_frame -
          int(total_frame*0.6) - int(total_frame*0.2))
    print("average #frame per video: ", total_frame//num_videos)

    PlotDataset(video_arr)


def PlotDataset(video_arr):
    i = 1
    for video in video_arr:
        frames = list(range(1, video.shape[0]+1))
        plt.plot(frames, video, label='Video' + str(i))
        i += 1

    # naming the x axis
    plt.xlabel('Number of frames')
    # naming the y axis
    plt.ylabel('Number of cubes')
    # giving a title to my graph
    # plt.title('Numbe ')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    # plt.show()
    # plt.savefig('figures/loss-{}.png'.format(exp_name))
    plt.savefig('results/no_spike.png')
    plt.clf()


if __name__ == '__main__':
    dataset_statistics()
