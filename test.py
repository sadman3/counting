from utils import LoadData
import tensorflow as tf


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


if __name__ == '__main__':
    Test("checkpoint")
