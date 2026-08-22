# target 1, base 7

python3 poison_attacks.py -file_name model_mnist_lr0.0075 -attack fc -base_class 7 -target_class 1 -poison_num 10 -epsilon 0.10 -step_size 0.01 -iterations 1000 -watermark_opacity 0.1
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: MNIST

--------------------------------------
Feature Collision Poisoning Attack Implementation


Parameters:
        Base Class: 7;
        Target Class: 1;
        Poison Budget: 10;
        Step Size: 0.01;
        Iterations: 1000;
        Epsilon: 0.1;
        Watermark Opacity: 0.1

Crafting FC Poisons: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 562.75it/s, loss=6.55]
        Saved figure to ./images/feature_collision_poisons/mnist_clean_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poisoned_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poison_perturbations.png
Original Prediction: 1
Clean Model Accuracy: 0.9266
Training DNN model with lr 0.1: 100%|█████████████████████████████████████████████████████████████████████| 50/50 [00:42<00:00,  1.17it/s, Train acc=0.931, Train loss=0.248, Val acc=0, Val loss=0]
Poisoned Prediction: 1; Success: False
Poisoned Model Accuracy: 0.9173

# Target 7, Base 1

python3 poison_attacks.py -file_name model_mnist_lr0.0075 -attack fc -base_class 1 -target_class 7 -poison_num 30 -epsilon 0.10 -step_size 0.01 -iterations 1000 -watermark_opacity 0.1
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: MNIST

--------------------------------------
Feature Collision Poisoning Attack Implementation


Parameters:
        Base Class: 1;
        Target Class: 7;
        Poison Budget: 30;
        Step Size: 0.01;
        Iterations: 1000;
        Epsilon: 0.1;
        Watermark Opacity: 0.1

Crafting FC Poisons: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 568.67it/s, loss=3.95]
        Saved figure to ./images/feature_collision_poisons/mnist_clean_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poisoned_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poison_perturbations.png
Original Prediction: 7
Clean Model Accuracy: 0.9266
Training DNN model with lr 0.1: 100%|██████████████████████████████████████████████████████████████████████| 50/50 [00:42<00:00,  1.19it/s, Train acc=0.93, Train loss=0.257, Val acc=0, Val loss=0]
Poisoned Prediction: 7; Success: False
Poisoned Model Accuracy: 0.9249


# Target 3, Base 8

python3 poison_attacks.py -file_name model_mnist_lr0.0075 -attack fc -base_class 8 -target_class 3 -poison_num 50 -epsilon 0.10 -step_size 0.01 -iterations 2000 -watermark_opacity 0.1
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: MNIST

--------------------------------------
Feature Collision Poisoning Attack Implementation


Parameters:
        Base Class: 8;
        Target Class: 3;
        Poison Budget: 50;
        Step Size: 0.01;
        Iterations: 2000;
        Epsilon: 0.1;
        Watermark Opacity: 0.1

Crafting FC Poisons: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:03<00:00, 560.02it/s, loss=2.05]
        Saved figure to ./images/feature_collision_poisons/mnist_clean_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poisoned_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poison_perturbations.png
Original Prediction: 3
Clean Model Accuracy: 0.9266
Training DNN model with lr 0.1: 100%|███████████████████████████████████████████████████████████████████████| 50/50 [00:43<00:00,  1.14it/s, Train acc=0.93, Train loss=0.25, Val acc=0, Val loss=0]
Poisoned Prediction: 3; Success: False
Poisoned Model Accuracy: 0.9238


# target 8, base 3

python3 poison_attacks.py -file_name model_mnist_lr0.0075 -attack fc -base_class 3 -target_class 8 -poison_num 30 -epsilon 0.20 -step_size 0.02 -iterations 1000 -watermark_opacity 0.15
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: MNIST

--------------------------------------
Feature Collision Poisoning Attack Implementation


Parameters:
        Base Class: 3;
        Target Class: 8;
        Poison Budget: 30;
        Step Size: 0.02;
        Iterations: 1000;
        Epsilon: 0.2;
        Watermark Opacity: 0.15

Crafting FC Poisons: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:02<00:00, 495.48it/s, loss=0.831]
        Saved figure to ./images/feature_collision_poisons/mnist_clean_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poisoned_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poison_perturbations.png
Original Prediction: 8
Clean Model Accuracy: 0.9266
Training DNN model with lr 0.1: 100%|█████████████████████████████████████████████████████████████████████| 50/50 [00:41<00:00,  1.21it/s, Train acc=0.929, Train loss=0.257, Val acc=0, Val loss=0]
Poisoned Prediction: 8; Success: False
Poisoned Model Accuracy: 0.9226

# target 9, base 4

python3 poison_attacks.py -file_name model_mnist_lr0.0075 -attack fc -base_class 4 -target_class 9 -poison_num 30 -epsilon 0.20 -step_size 0.02 -iterations 1000 -watermark_opacity 0.15
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: MNIST

--------------------------------------
Feature Collision Poisoning Attack Implementation


Parameters:
        Base Class: 4;
        Target Class: 9;
        Poison Budget: 30;
        Step Size: 0.02;
        Iterations: 1000;
        Epsilon: 0.2;
        Watermark Opacity: 0.15

Crafting FC Poisons: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 577.40it/s, loss=0.774]
        Saved figure to ./images/feature_collision_poisons/mnist_clean_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poisoned_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poison_perturbations.png
Original Prediction: 9
Clean Model Accuracy: 0.9266
Training DNN model with lr 0.1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:36<00:00,  1.37it/s, Train acc=0.931, Train loss=0.251, Val acc=0, Val loss=0]
Poisoned Prediction: 9; Success: False
Poisoned Model Accuracy: 0.9166


# target 6, base 5

python3 poison_attacks.py -file_name model_mnist_lr0.0075 -attack fc -base_class 5 -target_class 6 -poison_num 30 -epsilon 0.20 -step_size 0.02 -iterations 1000 -watermark_opacity 0.15
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: MNIST

--------------------------------------
Feature Collision Poisoning Attack Implementation


Parameters:
        Base Class: 5;
        Target Class: 6;
        Poison Budget: 30;
        Step Size: 0.02;
        Iterations: 1000;
        Epsilon: 0.2;
        Watermark Opacity: 0.15

Crafting FC Poisons: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 539.07it/s, loss=1.59]
        Saved figure to ./images/feature_collision_poisons/mnist_clean_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poisoned_images.png
        Saved figure to ./images/feature_collision_poisons/mnist_poison_perturbations.png
Original Prediction: 6
Clean Model Accuracy: 0.9266
Training DNN model with lr 0.1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:39<00:00,  1.27it/s, Train acc=0.929, Train loss=0.258, Val acc=0, Val loss=0]
Poisoned Prediction: 6; Success: False
Poisoned Model Accuracy: 0.9195