import argparse
import copy

import matplotlib.pyplot as plt
import torch

from poisoning import dataset
from poisoning.neural_network import NeuralNetwork, get_torch_device
from poisoning.plotting import plot_images
from poisoning.poison_crafting import craft_fc_poisons


def get_images_from_label(data, label, num):
    """ gets num images from class label. Returns images and indices """
    indices = []
    counter = 0
    for idx, (sample, sample_label) in enumerate(zip(*data)):
        if sample_label.item() == label:
            indices.append(idx)
            counter += 1
            if counter == num:
                break

    return data[0][indices], indices


def main(model, data, dataset_name):
    # feature collision settings
    step_size = 0.01
    iterations = 2000
    epsilon = 0.03
    watermark_opacity = 0.3

    # base class label
    base_class = 1
    # target class label
    target_class = 0
    # poison budget - number of poison samples
    poison_num = 20
    
    # get images  to use as bases for poisons and their indices from training data
    base_imgs, base_indices = get_images_from_label(data.get_train_data(), label=base_class, num=poison_num)
    # get image to use as target and its index from  data
    target_img, target_index = get_images_from_label(data.get_test_data(), label=target_class, num=1)

    
    # keep poison-crafting tensors on the same device as the model
    base_imgs = base_imgs.to(model.device)
    target_img = target_img.to(model.device)

    # get poison perturbations using FC attack
    delta = craft_fc_poisons(model, base_imgs, target_img, step_size, iterations=iterations, epsilon=epsilon,
                             watermark_opacity=watermark_opacity)
    delta_on_device = delta.to(model.device)

    # plot clean and poisoned images
    plot_images(data.unnormalize_data(base_imgs), 4, 8, data.is_grayscale(), title="Clean Images", filename=f"./images/{dataset_name}_clean_images.png")
    plot_images(data.unnormalize_data(base_imgs + delta_on_device), 4, 8, data.is_grayscale(), title="Poisoned Images", filename=f"./images/{dataset_name}_poisoned_images.png")

    # plot perturbations
    plot_images(data.unnormalize_data(delta_on_device), 4, 8, data.is_grayscale(), title="Poison Perturbations", filename=f"./images/{dataset_name}_poison_perturbations.png", isPerturbation=True)

    print(f'Original Prediction: {model.predict(target_img).item()}')
    print(f'Clean Model Accuracy: {model.test(data.testloader)}')

    # get copy of model with new output layer for finetuning
    poisoned_model = model.from_pretrained(1 if data.class_num == 2 else data.class_num)
    # add poison perturbation to base samples in the training set
    data.poison_data(delta, base_indices)

    # freeze parameters of the pretrained model, training only the new output layer
    poisoned_model.freeze_extractor(True)
    # train model
    poisoned_model.fit(data.trainloader, None, 0.1, 50)
    predicted = poisoned_model.predict(target_img).item()
    print(f'Poisoned Prediction: {predicted}; Success: {predicted == base_class}')
    print(f'Poisoned Model Accuracy: {poisoned_model.test(data.testloader)}')


def get_data_and_model(file_name, dataset_name):
    device = get_torch_device()
    # get dataset and move data to correct device (cpu or cuda)
    data = dataset.get_dataset(dataset_name=dataset_name, bs=128)
    # used to store training loss and accuracy for each learning rate used
    model = NeuralNetwork.load(file_name)
    model.to(device)
    return model, data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_name', help='model file name', required=True)
    args = parser.parse_args()
    
    # get dataset name from the name of the model
    dataset_name = args.file_name.split('_')[1]

    model, data = get_data_and_model(args.file_name, dataset_name)
    
    main(model, data, dataset_name)
