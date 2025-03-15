"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(".."))
import os
import torch
from torchvision import transforms
from src import data_setup, engine, model_builder, utils


# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "./inputs/pizza_steak_sushi/train"
test_dir = "./inputs/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
dataloader = data_setup.CustomDataloader(train_dir=train_dir, test_dir=test_dir,
                                         transform=data_transform, batch_size=BATCH_SIZE, transform_test_data=False)
train_dataloader, test_dataloader = dataloader.get_dataloaders()
class_names = dataloader.classes

# dataloader, class_names

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# # Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# # Start training with help from engine.py
results=engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device, log_interval=1)
engine.plot_loss_curves(results)
