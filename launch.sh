#!/bin/bash


# MANDATORY PARAMETERS

# model_file='model_cat_lr0.0075'
# model_file='model_mnist_lr0.0075'


attack_type='fc'
# attack_type='polytope'
# attack_type='gradient'


# ATTACK PARAMETERS
base=3
target=8

poison_num=50

epsilon=0.05
step_size=0.01
iterations=2000

watermark_opacity=0.1



# ATTACK  



## attack with only mandatory parameters
# echo "python3 poison_attacks.py -file_name $model_file -attack $attack_type"
# python3 poison_attacks.py -file_name $model_file -attack $attack_type

## attack with only mandatory parameters and base and target classes
# echo "python3 poison_attacks.py -file_name $model_file -attack $attack_type -base_class $base -target_class $target"
# python3 poison_attacks.py -file_name $model_file -attack $attack_type -base_class $base -target_class $target


if attack_type == 'fc'; then

    # attack complete of all parameters (FC)
    echo "python3 poison_attacks.py -file_name $model_file -attack $attack_type -base_class $base -target_class $target -poison_num $poison_num -epsilon $epsilon -step_size $step_size -iterations $iterations -watermark_opacity $watermark_opacity"
    python3 poison_attacks.py -file_name $model_file -attack $attack_type -base_class $base -target_class $target -poison_num $poison_num -epsilon $epsilon -step_size $step_size -iterations $iterations -watermark_opacity $watermark_opacity

else attack_type == 'polytope' || attack_type == 'gradient'; then

    # attack complete of all parameters (Polytope and Gradient)
    echo "python3 poison_attacks.py -file_name $model_file -attack $attack_type -base_class $base -target_class $target -poison_num $poison_num -epsilon $epsilon -step_size $step_size -iterations $iterations"
    python3 poison_attacks.py -file_name $model_file -attack $attack_type -base_class $base -target_class $target -poison_num $poison_num -epsilon $epsilon -step_size $step_size -iterations $iterations

fi

