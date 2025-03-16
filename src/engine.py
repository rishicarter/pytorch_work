import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

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

def plot_confusion_matrix_model(model: torch.nn.Module, dataloader, device: str="cpu", infer_test_data: bool=True):
    """
    Plots the confusion matrix for a given model using predictions from a test dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model.
    dataloader : data_step.CustomDataloader
        The dataset loader class present in data_step.py file.
    device : str, optional
        The device to use for computation ("cuda" or "cpu"), default is "cpu".
    infer_test_data: bool, optional
        Inference mode with Testing data or Training data. Model predicts on dataset based on this flag.

    Raises
    ------
    ValueError
        If the dataloader does not provide test data.
        If the model is not an instance of `torch.nn.Module`.
    """

    # Validate inputs
    if not isinstance(model, torch.nn.Module):
        raise ValueError("The provided model must be an instance of `torch.nn.Module`.")

    if not hasattr(dataloader, "test_dataloader"):
        raise ValueError("The provided dataloader does not contain a `test_dataloader` method or attribute.")

    # Set model to evaluation mode
    model.to(device)
    model.eval()

    y_preds, y_true = [], []

    # Disable gradient computation for inference
    with torch.inference_mode():
        for batch in dataloader.test_dataloader if infer_test_data else dataloader.train_dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Forward pass
            y_logits = model(x)

            # Compute predictions
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

            y_preds.append(y_pred)
            y_true.append(y)

    # Convert lists to tensors
    y_pred_tensor = torch.cat(y_preds).cpu()
    y_true_tensor = torch.cat(y_true).cpu()

    # Check if `dataloader` has `classes` attribute
    if not hasattr(dataloader, "classes"):
        raise ValueError("Dataloader does not have `classes` attribute defining class names.")

    class_names = dataloader.classes
    num_classes = len(class_names)

    # Compute confusion matrix
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    confmat_tensor = confmat(preds=y_pred_tensor, target=y_true_tensor)

    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(), class_names=class_names, figsize=(10, 7))
    plt.title(f"ConfMat Class To IDX - {dataloader.class_to_idx}")
    plt.show()

# def plot_predictions(model: torch.nn.Module, dataloader, device: str="cpu", infer_test_data: bool=True):

#     if infer_test_data:
#         data_len = dataloader.test_samples
#         plot_data = dataloader.test_dataloader
#     else:
#         data_len = dataloader.train_samples
#         plot_data = dataloader.train_dataloader
#     plt.figure(figsize=(15,7))
#     i=1
#     # Get random sample indexes
#     random_samples_idx = random.sample(range(len(self.train_data)), k=6)
#     for i, sample_idx in enumerate(random_samples_idx):
#         image, label = self.train_data[sample_idx]

#         # Adjust tensor shape for plotting: [C, H, W] -> [H, W, C]
#         image_adjusted = image.permute(1, 2, 0)

#         # Create subplot
#         plt.subplot(1, n, i + 1)
#         plt.imshow(image_adjusted)
#         plt.axis("off")

#         # Set title
#         title = f"Class: {self.classes[label]}" if self.classes else f"Label: {label}"
#         if display_shape:
#             title += f"\nShape: {tuple(image_adjusted.shape)}"

#         plt.title(title, fontsize=10)

#     # Display the images
#     plt.show()
#     while i!=7:
#         rand_idx = torch.randint(0, data_len, size=(1,)).item()
#         img, label = train_data[rand_idx]
#         rand_img = img.unsqueeze(0)
#         prediction = f_model(rand_img.to(DEVICE))
#         pred_ = torch.softmax(prediction.squeeze(), dim=0).argmax()
#         if pred_ != label:
#             plt.subplot(2, 3, i)
#             i+=1
#             plt.imshow(img.squeeze(), cmap="grey")
#             plt.title(f"Actual: {label} | Pred: {pred_.cpu().numpy()}")
#             plt.axis(False);
