import matplotlib.pyplot as plot

def print_progress(epoch, loss, accuracy, totalEpochs):
    percentage = round(100 * (epoch / totalEpochs))
    if epoch == 1:
        print('================')
    print(
        "Epoch: {}, ".format(epoch) +
        "Loss: {}, ".format(round(loss, 4)) +
        "Accuracy: {}".format(round(accuracy, 4)) +
        "\n{}% finished".format(percentage)
        )
    print('================')

def showLossGraph(lossList, numEpochs, label='Loss'):
    xList = range(1, numEpochs + 1)
    plot.scatter(xList, lossList)
    plot.title('Loss as a function of time')
    plot.xlabel('Epoch Number')
    plot.ylabel(label)
    plot.show()
