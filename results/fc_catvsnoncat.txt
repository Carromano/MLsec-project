./launch.sh 
Launching attack
python3 poison_attacks.py -file_name model_cat_lr0.0075 -attack fc -base_class 1 -target_class 0 -poison_num 30 -epsilon 0.03 -step_size 0.01 -iterations 2000 -watermark_opacity 0.2
Using CUDA device for hardware acceleration

--------------------------------------
Dataset in use: catvsnoncat

--------------------------------------
Feature Collision Poisoning Attack Implementation


Parameters:
        Base Class: 1;
        Target Class: 0;
        Poison Budget: 30;
        Step Size: 0.01;
        Iterations: 2000;
        Epsilon: 0.03;
        Watermark Opacity: 0.2

Crafting FC Poisons: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:03<00:00, 603.09it/s, loss=0.622]
        Saved figure to ./images/feature_collision_poisons/cat_clean_images.png
        Saved figure to ./images/feature_collision_poisons/cat_poisoned_images.png
        Saved figure to ./images/feature_collision_poisons/cat_poison_perturbations.png
Original Prediction: 0.0
Clean Model Accuracy: 0.74
Training DNN model with lr 0.1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 84.59it/s, Train acc=0.964, Train loss=0.113, Val acc=0, Val loss=0]
Poisoned Prediction: 1.0; Success: True
Poisoned Model Accuracy: 0.72