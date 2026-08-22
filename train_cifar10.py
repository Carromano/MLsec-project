import torch
from poisoning.dataset import get_dataset
from poisoning.neural_network import NeuralNetwork

def train_and_freeze_cifar10():
    dataset_name = "cifar10"
    
    # Loading the dataset from the torchvision library
    print(f"Loading {dataset_name} dataset...")
    data = get_dataset(dataset_name, bs=128, normalize_to_mean_std=True)
    
    # reshape the data to match the input shape expected by the model
    shape = [data.flattened_shape, 1024, 256, 128, data.class_num]
    activations = ['relu', 'relu', 'relu', 'identity']
    
    print("Neural Network Initialization...")
    model = NeuralNetwork(shape=shape, activations=activations, optimizer=torch.optim.Adam)
    
    # Training the model on the CIFAR-10 dataset and freezing it for later use in poisoning attacks
    print("Starting training...")
    epochs = 100
    lr = 0.0005
    
    # Train the model and validate it every 5 epochs
    history = model.fit(trainloader=data.trainloader, valloader=data.testloader, lr=lr, epochs=epochs, val_interval=5)
    
    # final evaluation on the test set
    final_acc = model.test(data.testloader)
    print(f"\nClean Accuracy on Test Set: {final_acc:.4f}")
    
    # Freezing the model to prevent further training and save it for later use in poisoning attacks
    file_name = "model_cifar10_lr0.0005"
    model.save(file_name)
    print(f"Model created, trained and saved as {file_name}")

if __name__ == '__main__':
    train_and_freeze_cifar10()