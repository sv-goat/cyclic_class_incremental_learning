import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset

from class_incremental_model import ClassIncrementalModel
from torch.utils.data import TensorDataset
import csv
import random
import os

def training_loop_baseline(
    input_size: int,
    num_classes: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    save_path: str,
    lr: float = 0.001,
    epochs: int = 5000,
    patience: int = 200,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ClassIncrementalModel(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    patience_counter = 0

    # Use 2e10 samples per class as the train dataloader
    # 100 samples per class in our test dataloader
    # validatino
    # Save train and test accuracy after every 100 epochs
    # Split into train dataloader into train and test dataloader
    train_data = train_dataloader.dataset

    # Convert dataset object into iterable
    # Split train data into train adn validation data
    new_train_size = int(len(train_data) * 0.8)
    train_data, validation_data = torch.utils.data.random_split(train_data, [new_train_size, len(train_data) - new_train_size])
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)

    # Make a tensor dataset
    indices = train_data.indices
    original_dataset = train_data.dataset
    train_data = TensorDataset(original_dataset.tensors[0][indices], original_dataset.tensors[1][indices])
    print("Length of train data", len(train_data))


    with open(save_path, 'a', newline='') as f:
        writer = csv.writer(f)

        model.train()
        best_loss = float('inf')
        for _ in range(epochs):
            # 1024 samples per class in train dataloder

            # Create new train dataloader with 1024 samples per class
            class_datasets = []  # List to hold datasets for each class
            for i in range(num_classes):
                # Assuming train_data has a method to get targets
                class_indices = [idx for idx, target in enumerate(train_data.tensors[1]) if target == i]

                # Sample 1024 points randomly
                random.shuffle(class_indices)
                sampled_indices = class_indices[:1024]

                class_dataset = Subset(train_data, sampled_indices)
                class_datasets.append(class_dataset)

            # Create the final dataloader
            train_dataloader = DataLoader(ConcatDataset(class_datasets), batch_size=64, shuffle=True)
            print("Length of train dataloadder", len(train_dataloader))
            epoch_loss = 0
            model.train()
            for inputs, labels in train_dataloader:
                # Training step
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_dataloader)
            print(f"Epoch [{_ + 1}/{epochs}], Loss: {epoch_loss}")

            # Validation step
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0
                for inputs, labels in validation_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_loss += criterion(outputs, labels).item()

                val_loss /= len(validation_dataloader)
                accuracy = 100 * correct / total
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    print("New best loss", best_loss)
                    torch.save(model.state_dict(), "best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        model.load_state_dict(torch.load("best_model.pt"))
                        break


        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            # Write the results to as a csv
            writer.writerow([accuracy, num_classes, lr])
        # Clear memory
        torch.cuda.empty_cache()
        del model
        del inputs, labels, outputs
