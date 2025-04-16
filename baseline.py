import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset, ConcatDataset
import random
from mnist1d.data import make_dataset, get_dataset_args
from trainin_loop_baseline import training_loop_baseline
import csv

if __name__ == '__main__':
    defaults = get_dataset_args()

    defaults.num_samples = 40000

    data = make_dataset(defaults)
    x, y, t = data['x'], data['y'], data['t']
    x_test, y_test = data['x_test'], data['y_test']
    
    # Print dataset info
    input_size = x.shape[1]
    num_classes = len(set(y))
    print("Input size", input_size)
    print("Number of classes", num_classes)

    # Convert to tensors
    x_train_tensor = torch.FloatTensor(np.stack([item[0] for item in x]))
    y_train_tensor = torch.LongTensor([item[1] for item in y])
    x_test_tensor = torch.FloatTensor(np.stack([item[0] for item in x_test]))
    y_test_tensor = torch.LongTensor([item[1] for item in y_test])

    # Create datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    num_classes_iter = [i+1 for i in range(1, 10)]
    learning_rate_iter = [1e-3, 1e-4, 1e-5, 1e-6]

    save_path = "results_baseline.csv"
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Accuracy", "Num classes", "Learning rate"])  # header row
    del f
    for num_classes in num_classes_iter:
        for lr in learning_rate_iter:
            print("Starting experiment with the following parameters")
            print(num_classes, lr)
            training_loop_baseline(input_size, num_classes, train_dataloader, test_dataloader, save_path, lr = lr)
