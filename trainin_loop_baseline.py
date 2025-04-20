import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from class_incremental_model import ClassIncrementalModel
from torch.utils.data import TensorDataset
import csv
import random

def training_loop_baseline(
    input_size: int,
    num_classes: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    save_path: str,
    lr: float = 0.001,
    epochs: int = 500,
    patience: int = 5,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ClassIncrementalModel(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with open(save_path, 'a', newline='') as f:
        writer = csv.writer(f)

        model.train()
        best_loss = float('inf')
        for _ in range(epochs):
            epoch_loss = 0
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
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
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
