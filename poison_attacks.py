import argparse
import copy

import matplotlib.pyplot as plt
import torch

from poisoning import dataset
from poisoning.neural_network import CIFARConvNet, NeuralNetwork, get_torch_device
from poisoning.plotting import plot_images
from poisoning.poison_crafting import craft_fc_poisons
from poisoning.poison_crafting import craft_polytope_poisons
from poisoning.poison_crafting import craft_gradient_matching_poisons


# Gets num images from class label. 
#   Returns images and indices
def get_images_from_label(data, label, num):
    indices = []
    counter = 0
    for idx, (sample, sample_label) in enumerate(zip(*data)):
        if sample_label.item() == label:
            indices.append(idx)
            counter += 1
            if counter == num:
                break

    return data[0][indices], indices


def main(model, data, dataset_name, attack_type, base_class, target_class, poison_num, epsilon=0.03, step_size=0.01, iterations=2000, watermark_opacity=0.3):

    # get images  to use as bases for poisons and their indices from training data
    base_imgs, base_indices = get_images_from_label(data.get_train_data(), label=base_class, num=poison_num)
    # get image to use as target and its index from  data
    target_img, target_index = get_images_from_label(data.get_test_data(), label=target_class, num=1)

    # keep poison-crafting tensors on the same device as the model
    base_imgs = base_imgs.to(model.device)
    target_img = target_img.to(model.device)       

    print("\n--------------------------------------")


    if dataset_name == "cat":

        print("Dataset in use: catvsnoncat") 

    elif dataset_name == "mnist":

        print("Dataset in use: MNIST")

    elif dataset_name == "cifar":

        print("Dataset in use: CIFAR-10")


    print("\n--------------------------------------")


    if attack_type == "fc":
        #--------------------------------------
        # Feature Collision Poisoning Attack
        #--------------------------------------

        print("Feature Collision Poisoning Attack Implementation\n")
    
        print(f"\nParameters:\n\tBase Class: {base_class};\n\tTarget Class: {target_class};\n\tPoison Budget: {poison_num};\n\tStep Size: {step_size};\n\tIterations: {iterations};\n\tEpsilon: {epsilon};\n\tWatermark Opacity: {watermark_opacity}\n")

        # get poison perturbations using FC attack
        delta = craft_fc_poisons(model, base_imgs, target_img, step_size, iterations=iterations, epsilon=epsilon, watermark_opacity=watermark_opacity)
        delta_on_device = delta.to(model.device)

        # plot clean and poisoned images
        plot_images(data.unnormalize_data(base_imgs), 4, 10, data.is_grayscale(), title="Clean Images", filename=f"./images/feature_collision_poisons/{dataset_name}_clean_images.png")
        plot_images(data.unnormalize_data(base_imgs + delta_on_device), 4, 10, data.is_grayscale(), title="Poisoned Images", filename=f"./images/feature_collision_poisons/{dataset_name}_poisoned_images.png")

        # plot perturbations
        plot_images(data.unnormalize_data(delta_on_device), 4, 10, data.is_grayscale(), title="Poison Perturbations", filename=f"./images/feature_collision_poisons/{dataset_name}_poison_perturbations.png", isPerturbation=True)

        # clean baseline evaluation
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

        # print the prediction of the poisoned model and whether it was successful in misclassifying the target image
        print(f'Poisoned Prediction: {predicted}; Success: {predicted == base_class}')
        print(f'Poisoned Model Accuracy: {poisoned_model.test(data.testloader)}')


    elif attack_type == "polytope":

        #--------------------------------------
        # Polytope Poisoning Attack
        #--------------------------------------

        print("Polytope Poisoning Attack Implementation\n")

        print(f"\nParameters:\n\tBase Class: {base_class};\n\tTarget Class: {target_class};\n\tPoison Budget: {poison_num};\n\tStep Size: {step_size};\n\tIterations: {iterations};\n\tEpsilon: {epsilon};\n\tWatermark Opacity: {watermark_opacity}\n")

        # get poison perturbations using POLYTOPE attack
        delta_polytope = craft_polytope_poisons(model, base_imgs, target_img, step_size, iterations=iterations, epsilon=epsilon)
        delta_on_device = delta_polytope.to(model.device)

        # plot clean and poisoned images
        plot_images(data.unnormalize_data(base_imgs), 4, 10, data.is_grayscale(), title="Clean Images", filename=f"./images/polytope_poisons/{dataset_name}_clean_images_polytope.png")
        plot_images(data.unnormalize_data(base_imgs + delta_on_device), 4, 10, data.is_grayscale(), title="Poisoned Images", filename=f"./images/polytope_poisons/{dataset_name}_poisoned_images_polytope.png")

        # plot perturbations
        plot_images(data.unnormalize_data(delta_on_device), 4, 10, data.is_grayscale(), title="Poison Perturbations", filename=f"./images/polytope_poisons/{dataset_name}_poison_perturbations_polytope.png", isPerturbation=True)

        # clean baseline evaluation
        print(f'Original Prediction: {model.predict(target_img).item()}')
        print(f'Clean Model Accuracy: {model.test(data.testloader)}')

        # get copy of model with new output layer for finetuning
        poisoned_model = model.from_pretrained(1 if data.class_num == 2 else data.class_num)

        # add poison perturbation to base samples in the training set
        data.poison_data(delta_polytope, base_indices)

        # freeze parameters of the pretrained model, training only the new output layer
        poisoned_model.freeze_extractor(True)

        # train model
        poisoned_model.fit(data.trainloader, None, 0.1, 50)
        predicted = poisoned_model.predict(target_img).item()

        # print the prediction of the poisoned model and whether it was successful in misclassifying the target image
        print(f'Poisoned Prediction: {predicted}; Success: {predicted == base_class}')
        print(f'Poisoned Model Accuracy: {poisoned_model.test(data.testloader)}')


    elif attack_type == "gradient":

        #--------------------------------------
        # Gradient Matching Poisoning Attack
        #--------------------------------------

        print("Gradient Matching Poisoning Attack Implementation\n")

        print(f"\nParameters:\n\tBase Class: {base_class};\n\tTarget Class: {target_class};\n\tPoison Budget: {poison_num};\n\tStep Size: {step_size};\n\tIterations: {iterations};\n\tEpsilon: {epsilon};\n\tWatermark Opacity: {watermark_opacity}\n")

        # get poison perturbations using GRADIENT MATCHING attack
        delta_gradient = craft_gradient_matching_poisons(model, base_imgs, target_img, step_size, iterations=iterations, epsilon=epsilon, watermark_opacity=watermark_opacity)
        delta_on_device = delta_gradient.to(model.device)

        # plot clean and poisoned images
        plot_images(data.unnormalize_data(base_imgs), 4, 10, data.is_grayscale(), title="Clean Images", filename=f"./images/gradient_matching/{dataset_name}_clean_images_gm.png")
        plot_images(data.unnormalize_data(base_imgs + delta_on_device), 4, 10, data.is_grayscale(), title="Poisoned Images", filename=f"./images/gradient_matching/{dataset_name}_poisoned_images_gm.png")

        # plot perturbations
        plot_images(data.unnormalize_data(delta_on_device), 4, 10, data.is_grayscale(), title="Poison Perturbations", filename=f"./images/gradient_matching/{dataset_name}_poison_perturbations_gm.png", isPerturbation=True)

        # clean baseline evaluation
        print(f'Original Prediction: {model.predict(target_img).item()}')
        print(f'Clean Model Accuracy: {model.test(data.testloader)}')

        # get copy of model with new output layer for finetuning
        poisoned_model = model.from_pretrained(1 if data.class_num == 2 else data.class_num)

        # add poison perturbation to base samples in the training set
        data.poison_data(delta_gradient, base_indices)

        # freeze parameters of the pretrained model, training only the new output layer
        poisoned_model.freeze_extractor(True)

        # train model
        poisoned_model.fit(data.trainloader, None, 0.1, 50)
        predicted = poisoned_model.predict(target_img).item()

        # print the prediction of the poisoned model and whether it was successful in misclassifying the target image
        print(f'Poisoned Prediction: {predicted}; Success: {predicted == base_class}')
        print(f'Poisoned Model Accuracy: {poisoned_model.test(data.testloader)}')

    else:

        # print error message for invalid attack type parameter
        print(f"Invalid attack type: {attack_type}. Please choose from 'fc', 'polytope', or 'gradient'.")

# function to get the dataset and model based on the provided file name and dataset name
def get_data_and_model(file_name, dataset_name):
    device = get_torch_device()
    # get dataset and move data to correct device (cpu or cuda)
    data = dataset.get_dataset(dataset_name=dataset_name, bs=128)
    # used to store training loss and accuracy for each learning rate used
    if dataset_name == "cifar10":
        model = CIFARConvNet(num_classes=data.class_num)
    else:
        model = NeuralNetwork.load(file_name)
    model.to(device)
    return model, data



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # MANDATORY parameters
    parser.add_argument('-file_name', help='model file name', required=True)
    parser.add_argument('-attack', help='attack type (fc, polytope, gradient)', required=True)

    # ATTACK parameters
    parser.add_argument('-base_class', type=int, default=1, help='Label for the base class')
    parser.add_argument('-target_class', type=int, default=0, help='Label for the target class')
    parser.add_argument('-poison_num', type=int, default=10, help='Number of poison samples to generate')
    parser.add_argument('-epsilon', type=float, default=0.03, help='Maximum perturbation allowed for each poison sample')
    parser.add_argument('-watermark_opacity', type=float, default=0.3, help='Opacity of the watermark applied to the poison samples')
    parser.add_argument('-step_size', type=float, default=0.01, help='Step size for the optimization process')
    parser.add_argument('-iterations', type=int, default=2000, help='Number of iterations for the optimization process')
    parser.add_argument('-lr', type=float, default=0.1, help='Learning rate for the optimization process')
    parser.add_argument('-epochs', type=int, default=50, help='Number of epochs for the optimization process')

    args = parser.parse_args()
    
    # get the model and data based on the provided file name and dataset name
    dataset_name = args.file_name.split('_')[1]
    model, data = get_data_and_model(args.file_name, dataset_name)

    # get the parameters from the command line arguments
    base_class = args.base_class
    target_class = args.target_class
    
    attack_type = args.attack.lower()
    poison_num = args.poison_num

    epsilon = args.epsilon
    step_size = args.step_size
    iterations = args.iterations
    watermark_opacity = args.watermark_opacity
    

    # launching the attack
    main(model, data, dataset_name, attack_type, base_class, target_class, poison_num, epsilon, step_size, iterations, watermark_opacity)
