import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from class_incremental_model import ClassIncrementalModel
import random
import numpy as np
from torch.utils.data import TensorDataset
import csv

def training_loop(input_size, num_classes, class_wise_data, class_dataloaders_test, save_path, lr = 0.001, num_samples = 32):
    # define the model and the parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ClassIncrementalModel(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # dataset parameter
    batch_size = 32
    max_cycles = 5000

    # define the training loop
    for cycle in range(max_cycles):

        # create dataset for each class
        class_datasets = {}
        for i in range(10):
            sampled_data = random.sample(class_wise_data[i],
                                         min(len(class_wise_data[i]), num_samples))  # Permute and limit to num_samples
            class_datasets[i] = TensorDataset(torch.stack([torch.FloatTensor(item[0]) for item in sampled_data]),
                                              torch.LongTensor([item[1] for item in sampled_data]))

        # create a dataloader for each class
        class_dataloaders = {}
        for i in range(10):
            class_dataloaders[i] = DataLoader(class_datasets[i], batch_size=batch_size, shuffle=True)

        for cla in range(num_classes):
            model.train()
            for inputs, labels in class_dataloaders[cla]:
                # Training step
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                post_cycle_accs = []
                for cla_test in range(num_classes):
                    correct, total = 0, 0
                    for inputs, labels in class_dataloaders_test[cla_test]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                    accuracy = 100 * correct / total
                    post_cycle_accs.append(accuracy)
                    # Write the results to as a csv
                    with open(save_path, 'a+', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([cycle + 1, post_cycle_accs, cla_test])
        if cycle % 500 == 0:
            print("Sanity")
            print(f"{cycle+1}, {accuracy} for class {cla_test}")
        # Clear memory
        torch.cuda.empty_cache()
        del inputs, labels, outputs
