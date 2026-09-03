import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from poisoning.dataset import get_dataset
from poisoning.neural_network import CIFARConvNet, NeuralNetwork, get_torch_device

# function to train and freeze a model on the CIFAR-10 dataset
def train_and_freeze_cifar(lr, epochs):
    device = get_torch_device()

    # data loading and preprocessing
    data = get_dataset('cifar10', bs=128, normalize_to_mean_std=False)    
    
    # Model initializationma 
    model = CIFARConvNet(num_classes=data.class_num)

    # splitting data into training and validation
    training = data.trainloader
    train_portion = int(0.8 * len(training.dataset))
    val_portion = len(training.dataset) - train_portion
    train_data, val_data = torch.utils.data.random_split(training.dataset, [train_portion, val_portion], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

    # training on clean CIFAR-10 dataset
    print(f"Starting training on CIFAR-10... (lr={lr}, epochs={epochs})")
    model.fit(train_loader, val_loader, lr=lr, epochs=epochs)
    
    # 4. Valutazione Clean Accuracy
    clean_acc = model.test(data.testloader)
    print(f"Baseline Clean Test Accuracy: {clean_acc:.3f}")
    
    # 5. Congelamento del Feature Extractor per Transfer Learning
    model.freeze_extractor()
    print("Feature Extractor frozen.")
    
    # 6. Salvataggio del Checkpoint
    os.makedirs("./models", exist_ok=True)
    save_path = f"./model_cifar10_lr{lr}_{epochs}epochs"
    model.save(save_path)
    print(f"Checkpoint saved with success in: {save_path}")


# function to train and freeze a model on the MNIST dataset
def train_and_freeze_mnist(lr, epochs):
    device = get_torch_device()

    data = get_dataset('mnist', bs=128, normalize_to_mean_std=False)    

    shape = [data.flattened_shape, 256, 128, data.class_num]
    activations = ["relu", "relu", "identity"]
    loss_fn = F.cross_entropy
    
    # input_dim: 28*28 = 784, output_dim: 10 classi
    model = NeuralNetwork(shape=shape, activations=activations, loss_fn=loss_fn)

    # splitting data
    training = data.trainloader
    train_portion = int(0.8 * len(training.dataset))
    val_portion = len(training.dataset) - train_portion
    train_data, val_data = torch.utils.data.random_split(training.dataset, [train_portion, val_portion], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

    # 3. Training su Clean Dataset
    print(f"Starting training on MNIST... (lr={lr}, epochs={epochs})")
    model.fit(train_loader, val_loader, lr=lr, epochs=epochs)
    
    # 4. Valutazione Clean Accuracy (Baseline)
    clean_acc = model.test(data.testloader)
    print(f"Baseline Clean Test Accuracy: {clean_acc:.3f}")
    
    # 5. Congelamento del Feature Extractor (Backbone)
    model.freeze_extractor(True)
    print("Feature Extractor frozen.")
    
    # 6. Salvataggio del Checkpoint
    os.makedirs("./models", exist_ok=True)
    save_path = f"./model_mnist_lr{lr}_{epochs}epochs"
    model.save(save_path)
    print(f"Checkpoint saved with success in: {save_path}\n")


# function to train and freeze a model on the Cat-vs-NonCat dataset
#    the validation set is actually disabled because the dataset used is too small and takes the model to overfit.
def train_and_freeze_cat(lr, epochs):
    device = get_torch_device()

    # dataset loading and preprocessing
    data = get_dataset('cat', bs=128)    
    
    # Model initialization
    shape = [data.flattened_shape, 32, 16, data.class_num]
    activations = ["relu", "relu", "identity"]
    loss_fn = F.cross_entropy

    model = NeuralNetwork(shape=shape, activations=activations, loss_fn=loss_fn, optimizer=torch.optim.Adam)

    # Data splitting into training and validation
    training = data.trainloader

    # Reshape the labels to be of type long for cross-entropy loss
    X, y = training.dataset.tensors
    training.dataset.tensors = (X, y.view(-1).long())

    # train_portion = int(0.8 * len(training.dataset))
    # val_portion = len(training.dataset) - train_portion
    # train_data, val_data = torch.utils.data.random_split(training.dataset, [train_portion, val_portion], generator=torch.Generator().manual_seed(42))

    train_data = training.dataset

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True) #, collate_fn=_cross_entropy_collate)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False) #, collate_fn=_cross_entropy_collate)

    
    # training on clean Cat-vs-NonCat dataset
    print(f"Starting training on Cat-vs-NonCat... (lr={lr}, epochs={epochs})")
    model.fit(train_loader, None, lr=lr, epochs=epochs, weight_decay=1e-2)

    # Reshape the test labels to be of type long for cross-entropy loss
    X_test, y_test = data.testloader.dataset.tensors
    data.testloader.dataset.tensors = (X_test, y_test.view(-1).long())
    
    # baseline clean accuracy evaluation
    clean_acc = model.test(data.testloader)
    print(f"Baseline Clean Test Accuracy: {clean_acc:.3f}")
    
    # freezing the feature extractor for transfer learning
    model.freeze_extractor(True)  # freeze all layers except the last one
    print("Feature Extractor frozen.")
    
    # checkpoint saving
    os.makedirs("./models", exist_ok=True)
    save_path = f"./model_cat_lr{lr}_{epochs}epochs"
    model.save(save_path)
    print(f"Checkpoint saved with success in: {save_path}\n")


# Main function to parse command line arguments and call the appropriate training function
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, default='cifar10', help='Dataset to use (default: cifar10)')
    parser.add_argument('-lr', type=float, default=0.075, help='Learning rate for training')
    parser.add_argument('-epochs', type=int, default=80, help='Number of epochs for training')

    args = parser.parse_args()

    dataset_name = args.dataset
    lr = args.lr
    epochs = args.epochs

    if dataset_name == 'cifar10':
        train_and_freeze_cifar(lr, epochs)
    elif dataset_name == 'mnist':
        train_and_freeze_mnist(lr, epochs)
    elif dataset_name == 'cat':
        train_and_freeze_cat(lr, epochs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are: cifar10, mnist, cat.")