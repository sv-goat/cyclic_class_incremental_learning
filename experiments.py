# imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset, ConcatDataset
import random
from mnist1d.data import make_dataset, get_dataset_args
from training_loop import training_loop

if __name__ == '__main__':
    defaults = get_dataset_args()

    defaults.num_samples = 40000

    data = make_dataset(defaults)
    x, y, t = data['x'], data['y'], data['t']
    x_test, y_test = data['x_test'], data['y_test']

    print(x.shape, y.shape)
    print(x_test.shape, y_test.shape)

    input_size = x.shape[1]
    num_classes = len(set(y))
    print("Input size", input_size)
    print("Number of classes", num_classes)

    # testing data

    class_wise_data_test = [[] for _ in range(num_classes)]  # Create 10 separate lists

    for i in range(len(y)):
        clas = int(y[i])
        class_wise_data_test[clas].append((x[i], clas))

    # create dataset for each class
    class_datasets_test = {}
    for i in range(10):
        sampled_data = class_wise_data_test[i]
        class_datasets_test[i] = TensorDataset(torch.stack([torch.FloatTensor(item[0]) for item in sampled_data]),
                                               torch.LongTensor([item[1] for item in sampled_data]))

    # create a dataloader for each class
    class_dataloaders_test = {}
    for i in range(10):
        class_dataloaders_test[i] = DataLoader(class_datasets_test[i], batch_size=32, shuffle=True)

    class_wise_data = [[] for _ in range(num_classes)]  # Create 10 separate lists

    for i in range(len(y)):
        clas = int(y[i])
        class_wise_data[clas].append((x[i], clas))

    num_classes_iter = [i+1 for i in range(1, 10)]
    learning_rate_iter = [1e-3, 1e-4, 1e-5, 1e-6]
    num_samples_iter = [32, 64, 128, 256, 512]

    save_path = "results.csv"

    for num_samples in num_samples_iter:
        for num_classes in num_classes_iter:
            for lr in learning_rate_iter:
                print("Starting experiment with the following parameters")
                print(num_samples, num_classes, lr)
                loss = training_loop(input_size, num_classes, class_wise_data, class_dataloaders_test, save_path, lr = lr, num_samples = num_samples)



