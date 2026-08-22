# target 0, base 1

python3 poison_attacks.py -file_name model_mnist_lr0.0075 -attack polytope -base_class 1 -target_class 0 -poison_num 50 -epsilon 0.10 -step_size 0.01 -iterations 2000 -watermark_opacity 0.0
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: MNIST

--------------------------------------
Polytope Poisoning Attack Implementation


Parameters:
        Base Class: 1;
        Target Class: 0;
        Poison Budget: 50;
        Step Size: 0.01;
        Iterations: 2000;
        Epsilon: 0.1;
        Watermark Opacity: 0.0

Crafting Polytope Poisons: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:03<00:00, 612.74it/s, loss=5.68]
        Saved figure to ./images/polytope_poisons/mnist_clean_images_polytope.png
        Saved figure to ./images/polytope_poisons/mnist_poisoned_images_polytope.png
        Saved figure to ./images/polytope_poisons/mnist_poison_perturbations_polytope.png
Original Prediction: 0
Clean Model Accuracy: 0.9266
Training DNN model with lr 0.1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:41<00:00,  1.20it/s, Train acc=0.93, Train loss=0.251, Val acc=0, Val loss=0]
Poisoned Prediction: 0; Success: False
Poisoned Model Accuracy: 0.9233

# target 7, base 1

python3 poison_attacks.py -file_name model_mnist_lr0.0075 -attack polytope -base_class 1 -target_class 7 -poison_num 50 -epsilon 0.05 -step_size 0.01 -iterations 2000 -watermark_opacity 0.1
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: MNIST

--------------------------------------
Polytope Poisoning Attack Implementation


Parameters:
        Base Class: 1;
        Target Class: 7;
        Poison Budget: 50;
        Step Size: 0.01;
        Iterations: 2000;
        Epsilon: 0.05;
        Watermark Opacity: 0.1

Crafting Polytope Poisons: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:03<00:00, 618.35it/s, loss=10.3]
        Saved figure to ./images/polytope_poisons/mnist_clean_images_polytope.png
        Saved figure to ./images/polytope_poisons/mnist_poisoned_images_polytope.png
        Saved figure to ./images/polytope_poisons/mnist_poison_perturbations_polytope.png
Original Prediction: 7
Clean Model Accuracy: 0.9266
Training DNN model with lr 0.1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:37<00:00,  1.32it/s, Train acc=0.927, Train loss=0.263, Val acc=0, Val loss=0]
Poisoned Prediction: 7; Success: False
Poisoned Model Accuracy: 0.9224

# target 3, base 8

python3 poison_attacks.py -file_name model_mnist_lr0.0075 -attack polytope -base_class 8 -target_class 3 -poison_num 50 -epsilon 0.05 -step_size 0.01 -iterations 2000 -watermark_opacity 0.1
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: MNIST

--------------------------------------
Polytope Poisoning Attack Implementation


Parameters:
        Base Class: 8;
        Target Class: 3;
        Poison Budget: 50;
        Step Size: 0.01;
        Iterations: 2000;
        Epsilon: 0.05;
        Watermark Opacity: 0.1

Crafting Polytope Poisons: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:03<00:00, 602.21it/s, loss=2.21]
        Saved figure to ./images/polytope_poisons/mnist_clean_images_polytope.png
        Saved figure to ./images/polytope_poisons/mnist_poisoned_images_polytope.png
        Saved figure to ./images/polytope_poisons/mnist_poison_perturbations_polytope.png
Original Prediction: 3
Clean Model Accuracy: 0.9266
Training DNN model with lr 0.1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:41<00:00,  1.21it/s, Train acc=0.93, Train loss=0.251, Val acc=0, Val loss=0]
Poisoned Prediction: 3; Success: False
Poisoned Model Accuracy: 0.8852


# target 8, base 3

python3 poison_attacks.py -file_name model_mnist_lr0.0075 -attack polytope -base_class 3 -target_class 8 -poison_num 50 -epsilon 0.05 -step_size 0.01 -iterations 2000 -watermark_opacity 0.1
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: MNIST

--------------------------------------
Polytope Poisoning Attack Implementation


Parameters:
        Base Class: 3;
        Target Class: 8;
        Poison Budget: 50;
        Step Size: 0.01;
        Iterations: 2000;
        Epsilon: 0.05;
        Watermark Opacity: 0.1

Crafting Polytope Poisons: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:03<00:00, 568.88it/s, loss=6.88]
        Saved figure to ./images/polytope_poisons/mnist_clean_images_polytope.png
        Saved figure to ./images/polytope_poisons/mnist_poisoned_images_polytope.png
        Saved figure to ./images/polytope_poisons/mnist_poison_perturbations_polytope.png
Original Prediction: 8
Clean Model Accuracy: 0.9266
Training DNN model with lr 0.1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:43<00:00,  1.16it/s, Train acc=0.93, Train loss=0.252, Val acc=0, Val loss=0]
Poisoned Prediction: 8; Success: False
Poisoned Model Accuracy: 0.9189

