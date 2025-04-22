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
    x_train_tensor = torch.FloatTensor(x)
    y_train_tensor = torch.LongTensor(y)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    num_classes_iter = [i+1 for i in range(1, 10)]
    learning_rate_iter = [1e-4]

    save_path = "results_baseline_5k.csv"
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Accuracy", "Num classes", "Learning rate"])  # header row
    del f
    for num_classes in num_classes_iter:
        # Filter data for current number of classes
        # Limit to 100 samples per class in testing.
        train_mask = y_train_tensor < num_classes
        
        # Create filtered datasets
        filtered_train_dataset = TensorDataset(
            x_train_tensor[train_mask],
            y_train_tensor[train_mask]
        )
        
        # Create filtered dataloaders
        filtered_train_dataloader = DataLoader(filtered_train_dataset, batch_size=64, shuffle=True)

        lr = 1e-4
        num_iters = 5

        for i in range(num_iters):

            # Limit to 100 samples per class in testing.
            test_datasets = []  # List to hold datasets for each class
            for i in range(num_classes):
                # Assuming train_data has a method to get targets
                class_indices = [idx for idx, target in enumerate(test_dataset.targets) if target == i]

                # Sample 1024 points randomly
                random.shuffle(class_indices)
                sampled_indices = class_indices[:100]

                class_dataset = Subset(test_dataset, sampled_indices)
                test_datasets.append(class_dataset)


            filtered_test_dataset = ConcatDataset(test_datasets)

            filtered_test_dataloader = DataLoader(filtered_test_dataset, batch_size=64, shuffle=False)




            print(f"\nStarting experiment with num_classes={num_classes}, lr={lr}")
            print(f"Training samples: {len(filtered_train_dataset)}")
            print(f"Test samples: {len(filtered_test_dataset)}")
            
            training_loop_baseline(
                input_size=input_size,
                num_classes=num_classes,
                train_dataloader=filtered_train_dataloader,
                test_dataloader=filtered_test_dataloader,
                save_path=save_path,
                lr=lr
            )
