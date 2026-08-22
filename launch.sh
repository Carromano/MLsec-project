#!/bin/bash


# MANDATORY PARAMETERS

# model_file='model_cat_lr0.0075'
model_file='model_mnist_lr0.0075'

attack_type='fc'
# attack_type='polytope'
# attack_type='gradient'



# OPTIONAL PARAMETERS
target=6
base=5


# ATTACK PARAMETERS
poison_num=30
epsilon=0.20
watermark_opacity=0.15
step_size=0.02
iterations=1000


# ATTACK COMMAND 

echo "Launching attack"

## attack with only mandatory parameters
# echo "python3 poison_attacks.py -file_name $model_file -attack $attack_type"
# python3 poison_attacks.py -file_name $model_file -attack $attack_type

## attack complete of all parameters
echo "python3 poison_attacks.py -file_name $model_file -attack $attack_type -base_class $base -target_class $target -poison_num $poison_num -epsilon $epsilon -step_size $step_size -iterations $iterations -watermark_opacity $watermark_opacity"
python3 poison_attacks.py -file_name $model_file -attack $attack_type -base_class $base -target_class $target -poison_num $poison_num -epsilon $epsilon -step_size $step_size -iterations $iterations -watermark_opacity $watermark_opacity
