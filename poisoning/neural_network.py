import copy
import itertools
import sys

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from poisoning import constants

act_dict = {'relu': nn.ReLU, 'lrelu': nn.LeakyReLU, 'prelu':nn.PReLU, 'identity': nn.Identity, 'sigmoid': nn.Sigmoid}

__torch_device = None

"""
Function to get the torch device (CPU, CUDA, MPS) to use.
"""
def get_torch_device():
    global __torch_device

    if __torch_device is not None:
        return __torch_device

    if  torch.backends.mps.is_available():
        print('Using MPS device for hardware acceleration')
        __torch_device = 'mps'
    elif torch.cuda.is_available():
        print('Using CUDA device for hardware acceleration')
        __torch_device = 'cuda'
    else:
        print('Hardware acceleration not available, using CPU')
        __torch_device = 'cpu'
    return __torch_device


"""
Neural Network class that implements a feedforward neural network with arbitrary number of layers and activation functions.
Taken from the last laboratory of the course.
"""
class NeuralNetwork(nn.Module):
    
    def __init__(self, shape, activations, loss_fn=None, optimizer=torch.optim.SGD):
        """
        @param shape: list with size of each layer
        @param activations: type of activation function for each layer (all neurons in a layer use same activation)
        @loss_fn: loss function to use
        @optim: optimization algorithm to use
        """
        super().__init__()

        self.shape = shape
        self.activations = activations
        self.is_binary = shape[-1] == 1
        self.device = get_torch_device()

        # create list of layers
        layer_list = [nn.Linear(x, y) for x, y in zip(shape[:-2], shape[1:-1])]

        # create list of activations
        activations = [act_dict[a]() for a in activations]

        # create list of layers alternating with activations: [layer1, act1, ..., layerN, actN]
        layer_act_list = list(itertools.chain.from_iterable(zip(layer_list, activations)))

        # save index of penultimate layer for feature extraction
        self.feature_extractor_layer = len(layer_act_list) -1

        # add last layer and, if output is binary, sigmoid at last layer to squash to [0,1] interval
        layer_act_list.append(nn.Linear(shape[-2], shape[-1]))
        if shape[-1] == 1:
            layer_act_list.append(nn.Sigmoid())

        # create sequential module. It chains all the layers and activations sequentially
        self.model = nn.Sequential(*layer_act_list).to(self.device)
        if loss_fn is None:
            self.loss_fn = F.binary_cross_entropy if self.is_binary else F.cross_entropy
        else:
            self.loss_fn = loss_fn

        # initialize optimizer
        self.optimizer_type = optimizer


    """ implement forward method by calling the forward method of the Sequential module self.model """
    def forward(self, x):
        return self.model(x)


    """ returns the feature representation at the penultimate layer (before classification layer) """
    def features(self, x):
        return self.model[:self.feature_extractor_layer+1](x)


    """ return a copy of the model with new output layer of shape out_features """
    def from_pretrained(self, out_features):
        # copy base model
        model_copy = copy.deepcopy(self)

        # update last layer with new output shape
        output_layer = nn.Linear(self.model[self.feature_extractor_layer+1].in_features, out_features)
        model_copy.model = torch.nn.Sequential(*self.model[:self.feature_extractor_layer+1], output_layer)

        # add sigmoid if binary classification
        if out_features == 1:
            model_copy.model.add_module(f'{len(model_copy.model)}', nn.Sigmoid())

        return model_copy.to(self.device)


    """ transform model's outputs into binary predictions (if model is binary classifier)
        or index of highest probability class (argmax) otherwise """
    def _pred(self, outputs):
        if self.is_binary:
            return (outputs >= 0.5) * 1.
        else:
            return torch.argmax(outputs, dim=1)


    """ predict labels. 0 or 1 if binary, top class (argmax) if not """
    def predict(self, x):
        x = x.to(self.device)
        return self._pred(self(x))


    """ freeze the feature extractor (all layers up to penultimate layer) """
    def freeze_extractor(self, flag):
        for param in self.model[:self.feature_extractor_layer+1].parameters():
            # set require_grad to false if finetuning (flag==True)
            param.requires_grad = not flag


    def fit(self, trainloader, valloader, lr, epochs, val_interval=5, weight_decay=1e-5):
        """ fit (train) the model
        @param trainloader: training set in form of dataloader
        @param valloader: validation set in form of dataloader
        @param lr: learning rate
        @param epochs: number of iterations over the dataset to train
        @param val_interval: interval between output logs
        @return: loss and accuracy history
        """

        loss_history = {'train': [], 'val': []}
        acc_history = {'train': [], 'val': []}

        optimizer = self.optimizer_type(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.train()

        with tqdm(range(epochs), desc=f'Training DNN model with lr {lr}', file=sys.stdout) as pbar:
            val_acc, val_loss = 0., 0.
            for epoch in pbar:
                accuracy, running_loss = 0., 0.
                for x_batch, y_batch in trainloader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    # get predictions for the batch
                    pred = self(x_batch)
                    # compute loss
                    loss = self.loss_fn(pred, y_batch)
                    # backpropagate the loss and compute gradients using autograd
                    loss.backward()
                    # perform optimization step (update weights with computed gradients)
                    optimizer.step()
                    # zero-out gradients before next iteration
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    accuracy += (self._pred(pred) == y_batch).sum().item() / x_batch.shape[0]

                train_acc = accuracy / len(trainloader)
                train_loss = running_loss/len(trainloader)
                loss_history['train'].append((epoch, train_loss))
                acc_history['train'].append((epoch, train_acc))
                if (epoch + 1) % val_interval == 0 and valloader is not None:
                    val_acc, val_loss = self.test(valloader, compute_loss=True)
                    loss_history['val'].append((epoch, val_loss))
                    acc_history['val'].append((epoch, val_acc))
                pbar.set_postfix({'Train acc': train_acc,
                                  'Train loss': train_loss,
                                  'Val acc': val_acc,
                                  'Val loss': val_loss})
        return {'Loss': loss_history, 'Accuracy': acc_history}


    def test(self, testloader, compute_loss=False):
        correct, total, loss = 0., 0., 0.

        # we don't want to compute any gradiends here, we're just testing
        self.eval()
        with torch.no_grad():
            for x_batch, y_batch in testloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                # predict outputs
                pred = self(x_batch)

                # compute loss if compute_loss==True
                if compute_loss:
                    loss += self.loss_fn(pred, y_batch)

                # get number of correct predictions
                correct += (self._pred(pred) == y_batch).sum().item()
                total += x_batch.shape[0]

        # return accuracy and loss if requested, otherwise only accuracy
        accuracy = correct/total
        if compute_loss:
            return accuracy, (loss/len(testloader)).item()
        return accuracy

 
    def save(self, fname):
        constants.model_dir.mkdir(parents=True, exist_ok=True)
        to_save = {'shape': self.shape,
                   'activations': self.activations,
                   'loss': self.loss_fn,
                   'state_dict': self.state_dict()}
        torch.save(to_save, constants.model_dir / fname)


    @staticmethod
    def load(fname):
        device = get_torch_device()
        # weights_only = False needed to load old model
        saved = torch.load(constants.model_dir / fname, map_location=device, weights_only=False)
        model = NeuralNetwork(saved['shape'], saved['activations'], saved['loss'])
        model.load_state_dict(saved['state_dict'])
        return model


"""
Convolutional network compatible with the Transfer Learning wrapper used for the project.
Written specifically to test the attacks also on complex datasets like CIFAR10.
Exposes the same public interface as NeuralNetwork so that attack code can treat the
two architectures interchangeably
"""
class CIFARConvNet(nn.Module):

    def __init__(self, num_classes=10, loss_fn=None, optimizer=torch.optim.AdamW):
        """
        @param num_classes: number of output classes for the classification head
        @param loss_fn: loss function to use (defaults to cross entropy, as CIFARConvNet is always multi-class)
        @param optimizer: optimization algorithm to use
        """
        super().__init__()

        self.num_classes = num_classes
        self.is_binary = False
        self.device = get_torch_device()
        
        # Convolutional Backbone (phi) - feature extractor, equivalent role to NeuralNetwork's layers up to feature_extractor_layer
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.feature_dim = 128
        # Output Classifier Layer, equivalent role to NeuralNetwork's last Linear layer
        self.fc = nn.Linear(self.feature_dim, num_classes)

        self.model = nn.Sequential(self.extractor, self.fc).to(self.device)

        self.loss_fn = F.cross_entropy if loss_fn is None else loss_fn
        self.optimizer_type = optimizer


    """ return the features extracted by the backbone """
    def features(self, x):
        return self.extractor(x)


    """ implement forward method """
    def forward(self, x):
        feat = self.features(x)
        return self.fc(feat)

    """ Freeze the feature extractor for finetuning """
    def freeze_extractor(self, flag=True):
        for param in self.extractor.parameters():
            # set require_grad to false if finetuning (flag==True)
            param.requires_grad = not flag

    """ return a copy of the model with a new output layer of shape out_features """
    def from_pretrained(self, num_classes=10):
        self.freeze_extractor()
        self.fc = nn.Linear(self.feature_dim, num_classes).to(self.device)
        for param in self.fc.parameters():
            param.requires_grad = True
        return self

    """ index of highest probability class (argmax); CIFARConvNet is always multi-class """
    def _pred(self, outputs):
        return torch.argmax(outputs, dim=1)

    """ predict labels (top class / argmax) """
    def predict(self, x):
        x = x.to(self.device)
        return self._pred(self(x))


    def fit(self, trainloader, valloader, lr, epochs, val_interval=5):
        """ fit (train) the model
        @param trainloader: training set in form of dataloader
        @param valloader: validation set in form of dataloader
        @param lr: learning rate
        @param epochs: number of iterations over the dataset to train
        @param val_interval: interval between output logs
        @return: loss and accuracy history
        """

        loss_history = {'train': [], 'val': []}
        acc_history = {'train': [], 'val': []}

        # optimize only parameters that require grad (e.g. only the head if extractor is frozen)
        optimizer = self.optimizer_type(filter(lambda p: p.requires_grad, self.parameters()),
                                         lr=lr, weight_decay=1e-4)
        self.train()

        with tqdm(range(epochs), desc=f'Training CIFARConvNet model with lr {lr}', file=sys.stdout) as pbar:
            val_acc, val_loss = 0., 0.
            for epoch in pbar:
                accuracy, running_loss = 0., 0.
                for x_batch, y_batch in trainloader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    # get predictions for the batch
                    pred = self(x_batch)
                    # compute loss
                    loss = self.loss_fn(pred, y_batch)
                    # backpropagate the loss and compute gradients using autograd
                    loss.backward()
                    # perform optimization step (update weights with computed gradients)
                    optimizer.step()
                    # zero-out gradients before next iteration
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    accuracy += (self._pred(pred) == y_batch).sum().item() / x_batch.shape[0]

                train_acc = accuracy / len(trainloader)
                train_loss = running_loss / len(trainloader)
                loss_history['train'].append((epoch, train_loss))
                acc_history['train'].append((epoch, train_acc))
                if (epoch + 1) % val_interval == 0 and valloader is not None:
                    val_acc, val_loss = self.test(valloader, compute_loss=True)
                    loss_history['val'].append((epoch, val_loss))
                    acc_history['val'].append((epoch, val_acc))
                pbar.set_postfix({'Train acc': train_acc,
                                  'Train loss': train_loss,
                                  'Val acc': val_acc,
                                  'Val loss': val_loss})
        return {'Loss': loss_history, 'Accuracy': acc_history}


    def test(self, testloader, compute_loss=False):
        correct, total, loss = 0., 0., 0.

        self.eval()
        with torch.no_grad():
            for x_batch, y_batch in testloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                # predict outputs
                pred = self(x_batch)
                if compute_loss:
                    loss += self.loss_fn(pred, y_batch)

                # get number of correct predictions
                correct += (self._pred(pred) == y_batch).sum().item()
                total += x_batch.shape[0]

        # return accuracy and loss if requested, otherwise only accuracy
        accuracy = correct / total
        if compute_loss:
            return accuracy, (loss / len(testloader)).item()
        return accuracy

    def save(self, fname):
        constants.model_dir.mkdir(parents=True, exist_ok=True)
        to_save = {'num_classes': self.num_classes,
                   'loss': self.loss_fn,
                   'state_dict': self.state_dict()}
        torch.save(to_save, constants.model_dir / fname)

    @staticmethod
    def load(fname):
        device = get_torch_device()

        saved = torch.load(constants.model_dir / fname, map_location=device, weights_only=False)
        model = CIFARConvNet(saved['num_classes'], saved['loss'])
        model.load_state_dict(saved['state_dict'])

        return model