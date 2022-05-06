from tkinter import Frame
from utils import LoadData
import tensorflow as tf
import numpy as np


def Predict(model, testingData):
    # predict and format output to use with sklearn
    predict = model.predict(testingData)
    # predict = np.argmax(predict, axis=1)

    return predict


def Test(name):
    print("Loading Test Data")
    testingData, testingLabels = LoadData("test")

    print("Loading model")
    model = tf.keras.models.load_model("./models/"+name+".h5")
    print("Making predictions on test data")

    prediction = Predict(model, testingData)

    model.evaluate(testingData, testingLabels)

    f = open('results/prediction.txt', 'w')

    f.write("ground truth --- prediction\n")

    for i in range(prediction.shape[0]):
        f.write("{}             {}\n".format(
            testingLabels[i], prediction[i][0]))

    # prediction for entire video starts from here ------

    f.write("--------- video prediction ----------")

    print("\nMaking prediction for first and last frame\n")
    testingVideo = np.load("video/frames.npy")[..., 0:3]
    real_count = np.load("video/count.npy")
    prediction = Predict(model, testingVideo)

    model.evaluate(testingVideo, real_count)

    for i in range(prediction.shape[0]):
        print("video {}: real count: {} prediction: {}".format(
            (i // 2)+1, real_count[i], prediction[i][0]))

    mae = 0
    mse = 0

    los = 0

    print("computing for entire video")
    for i in range(0, len(prediction), 2):
        los += (prediction[i] - real_count[i])**2
        los += (prediction[i+1] - real_count[i+1])**2
        ground_truth = real_count[i] - real_count[i+1]
        predicted_count = prediction[i] - prediction[i+1]
        # f.write("{}             {}\n".format(
        #     real_count[i], final_precition[i]))

        print("real count:{}, predicted count:{}".format(
            ground_truth, predicted_count))

        mae += abs(ground_truth - predicted_count)
        mse += (ground_truth - predicted_count)**2

    mae /= 20.0
    mse /= 20.0

    print("los ", los/40)
    print("Mean Absoulute Error: ", mae)
    print("Mean Squared Error: ", mse)


if __name__ == '__main__':
    saved_model = "checkpoint"
    Test(saved_model)
