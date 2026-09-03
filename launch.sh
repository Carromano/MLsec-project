#!/bin/bash


# MANDATORY PARAMETERS
## CAT
model_file='model_cat_lr0.0075'


## MNIST
# model_file='model_mnist_lr0.0075'
# model_file='model_mnist_lr0.01_20epochs'
# model_file='model_mnist_lr0.05_20epochs'


## CIFAR
# model_file='./model_cifar10_lr0.001_30epochs'   


## ATTACKS
attack_type='fc'
# attack_type='polytope'
# attack_type='gradient'



# ATTACK PARAMETERS

base=1
target=0

poison_num=30

epsilon=0.01
step_size=0.01
iterations=2000

watermark_opacity=0.2

lr=0.1
epochs=20



# ATTACK  
echo "python3 poison_attacks.py -file_name $model_file -attack $attack_type -base_class $base -target_class $target -poison_num $poison_num -epsilon $epsilon -step_size $step_size -iterations $iterations -watermark_opacity $watermark_opacity"
echo ""
python3 poison_attacks.py -file_name $model_file -attack $attack_type -lr $lr -epochs $epochs -base_class $base -target_class $target -poison_num $poison_num -epsilon $epsilon -step_size $step_size -iterations $iterations -watermark_opacity $watermark_opacity