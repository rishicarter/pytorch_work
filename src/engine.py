import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               acc_fn: torchmetrics.Metric = None,
               device: str = "cpu"):
    """Performs one training step for a given model."""
    model.train()
    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Compute loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Compute accuracy
        y_pred_probs = torch.softmax(y_pred, dim=1)
        y_pred_labels = torch.argmax(y_pred_probs, dim=1)
        acc = (y_pred_labels == y).float().mean().item()

        train_acc += acc

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_step(model: nn.Module,
              dataloader: DataLoader,
              loss_fn: nn.Module,
              acc_fn: torchmetrics.Metric = None,
              device: str = "cpu"):
    """Performs one testing step for a given model."""
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)

            # Compute loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Compute accuracy
            y_pred_probs = torch.softmax(y_pred, dim=1)
            y_pred_labels = torch.argmax(y_pred_probs, dim=1)
            acc = (y_pred_labels == y).float().mean().item()

            test_acc += acc

    return test_loss / len(dataloader), test_acc / len(dataloader)


def train(model: nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module = nn.CrossEntropyLoss(),
          acc_fn: torchmetrics.Metric = None,
          epochs: int = 5,
          device: str = "cpu",
          log_interval: int = 1):
    """
    Trains and evaluates a given model.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    train_dataloader : DataLoader
        The DataLoader providing batches of training data.
    test_dataloader : DataLoader
        The DataLoader providing batches of testing data.
    optimizer : torch.optim.Optimizer
        The optimizer for updating model weights.
    loss_fn : nn.Module, optional
        The loss function (default is CrossEntropyLoss).
    acc_fn : torchmetrics.Metric, optional
        The accuracy metric function.
    epochs : int, optional
        The number of epochs to train (default is 5).
    device : str, optional
        The device to run computations on ("cpu" or "cuda", default is "cpu").
    log_interval : int, optional
        Logging interval (1 = every epoch, 5 = 1st, 5th, 10th, etc., 0 = no logs).

    Returns
    -------
    dict
        Dictionary containing training and testing loss/accuracy values.
    """
    model.to(device)
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, acc_fn, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, acc_fn, device)

        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Log based on log_interval
        if log_interval > 0 and (epoch == 0 or (epoch + 1) % log_interval == 0):
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return results

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.
    
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']
    
    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    
    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))
    
    # Setup a plot 
    plt.figure(figsize=(15, 7))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

    plt.show()
