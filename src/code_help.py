import matplotlib.pyplot as plt

def plot_predictions(X_train, y_train, X_test, y_test, predictions=None):
    '''
    PLots training data, test data and compares predictions
    '''
    plt.figure(figsize=(10,7))
    plt.scatter(X_train, y_train, c='b', s=4, label='Training data')
    plt.scatter(X_test, y_test, c='g', s=4, label='Testing data')

    if predictions is not None:
        plt.scatter(X_test, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={'size':14});
    

def plot_train_test_curves(epoch_count, train_losses, test_losses):
    plt.plot(epoch_count, train_losses, label="Train loss")
    plt.plot(epoch_count, test_losses, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend();