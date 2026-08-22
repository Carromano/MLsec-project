# correct parameters

python3 poison_attacks.py -file_name model_cat_lr0.0075 -attack polytope -base_class 1 -target_class 0 -poison_num 50 -epsilon 0.04 -step_size 0.005 -iterations 3000 -watermark_opacity 0.2
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: catvsnoncat

--------------------------------------
Polytope Poisoning Attack Implementation


Parameters:
        Base Class: 1;
        Target Class: 0;
        Poison Budget: 50;
        Step Size: 0.005;
        Iterations: 3000;
        Epsilon: 0.04;
        Watermark Opacity: 0.2

Crafting Polytope Poisons: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3000/3000 [00:04<00:00, 657.46it/s, loss=0.0849]
        Saved figure to ./images/polytope_poisons/cat_clean_images_polytope.png
        Saved figure to ./images/polytope_poisons/cat_poisoned_images_polytope.png
        Saved figure to ./images/polytope_poisons/cat_poison_perturbations_polytope.png
Original Prediction: 0.0
Clean Model Accuracy: 0.74
Training DNN model with lr 0.1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 81.97it/s, Train acc=0.954, Train loss=0.116, Val acc=0, Val loss=0]
Poisoned Prediction: 1.0; Success: True
Poisoned Model Accuracy: 0.72




# totally wrong parameters

python3 poison_attacks.py -file_name model_cat_lr0.0075 -attack polytope -base_class 1 -target_class 0 -poison_num 15 -epsilon 0.4 -step_size 0.5 -iterations 500 -watermark_opacity 0.5
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: catvsnoncat

--------------------------------------
Polytope Poisoning Attack Implementation


Parameters:
        Base Class: 1;
        Target Class: 0;
        Poison Budget: 15;
        Step Size: 0.5;
        Iterations: 500;
        Epsilon: 0.4;
        Watermark Opacity: 0.5

Crafting Polytope Poisons: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 543.22it/s, loss=140]
        Saved figure to ./images/polytope_poisons/cat_clean_images_polytope.png
        Saved figure to ./images/polytope_poisons/cat_poisoned_images_polytope.png
        Saved figure to ./images/polytope_poisons/cat_poison_perturbations_polytope.png
Original Prediction: 0.0
Clean Model Accuracy: 0.74
Training DNN model with lr 0.1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 96.08it/s, Train acc=0.996, Train loss=0.0316, Val acc=0, Val loss=0]
Poisoned Prediction: 0.0; Success: False
Poisoned Model Accuracy: 0.7