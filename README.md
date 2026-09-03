# MLsec-project
## Clean-Label Poisoning: From Feature Space Constraints to Gradient Matching

This project explores and evaluates the evolution of Clean-Label Data Poisoning attacks against Neural Networks within a Transfer Learning threat model. In this scenario, the attacker cannot alter the labels of the dataset, and the target model acts as a fixed feature extractor while only the final layer is retrained. The research compares the effectiveness, stealthiness, and transferability of three distinct poisoning methodologies using the `catvnoncat`, `MNIST` and `CIFAR-10` datasets. The final goal is to try to mislead the model into misclassifying a specific target instance by injecting a small number of crafted poison samples into the training dataset.

Methodologies Evaluated:  
  1. **Feature Collision:** A targeted approach that minimizes the $L_2$ Euclidean distance between the poisoned sample and the target instance within the model's feature space. To ensure visual stealthiness, the perturbation is strictly bounded by an $L_\infty$ norm applied directly in the input space.
  2. **Convex Polytope (Bullseye):** A multi-poison strategy that traps the target instance inside a convex hull formed by the poisons in the feature space. By assigning equal weights ($1/k$) to the distribution coefficients, the target is perfectly centered (Bullseye). This structural relaxation drastically improves the attack's transferability to unknown black-box models.
  3. **Gradient Matching:** An optimization technique that shifts the focus from feature space representations to the learning dynamics. It generates poisons by minimizing the negative cosine similarity between their loss gradients and the target's gradients ($\nabla_\theta \mathcal{L}$). This alignment effectively hijacks the gradient descent direction during the final layer retraining.


<br>

## Requirements

```txt
# Python

Python ≥ 3.9

```

```sh
# External libraries
pip install torch torchvision torchaudio numpy matplotlib

# or

pip install -r requirements.txt
```


<br>


## Project structure

```text
MLsec-project/
├── poison_attacks.py                  # Main attack launcher and experiment runner.
├── training.py                        # Script for training and freezing models.
├── launch.sh                          # Bash wrapper for launching attacks with parametric python commands.
├── train.sh                           # Bash wrapper for launching model training with parametric python commands.
├── asr_compute.sh                     # Bash script for evaluating the Attack Success Rate (ASR) of the attacks.
├── asr_analyze.py                       # Script for analyzing the ASR results and generating a summary report.
|
├── report/                            # Folder containing the generated reports for the ASR
│   ├── results.csv                    # massive testing results
│   ├── ASR.csv                        # ASR evaluations grouped by parameters combination
│   └── ASR_overall.csv                # ASR evaluations grouped by model and attack type
│
├── poisoning/                         # Dataset loading and poisoning utilities.
│   ├── constants.py                   # Shared configuration constants.
│   ├── dataset.py                     # Dataset loading and preprocessing helpers.
│   ├── neural_network.py              # Neural-network architecture and training logic.
│   ├── plotting.py                    # Visualization utilities for attacks and results.
│   └── poison_crafting.py             # Poison generation for the evaluated attacks.
│
├── data/                              # Datasets used by the project.
│   ├── test_catvnoncat.h5             # Cat-vs-non-cat test split.
│   ├── train_catvnoncat.h5            # Cat-vs-non-cat training split.
│   ├── cifar-10-python.tar.gz         # CIFAR-10 archive - not present on the repository due to size constraints
│   └── MNIST/                         # MNIST train/test raw files.
│       └── raw/
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
│
├── images/                            # Visualizations and plots generated during the experiments. 
│   ├── feature_collision_poisons/      # Feature Collision attack visualizations.
│   ├── gradient_matching/              # Gradient Matching attack visualizations.
│   └── polytope_poisons/               # Bullseye Polytope attack visualizations.
│
├── models/                            # Pre-trained and frozen model artifacts
│   └── model_DATASET_lr0.XXX_YYepochs  # Naming pattern for pre-trained models: dataset name, 
|                                       #     learning rate, and number of epochs used for training.
│                                       # The only two exceptions are the models provided by the course laboratory:
│                                       #       model_cat_lr0.0075 and model_mnist_lr0.0075
│
├── requirements.txt                   # Python dependencies for the project.
├── README.md                          # Project documentation and report
├── LICENSE                            # GitHub repository license.
└── .gitignore                         # Repository ignore rules.
```

<br>
<br>


# Training the models

From the root project directory, the baseline models can be trained and prepared using the following command:

`python3 training.py -dataset <dataset_name>`

Note that:
  - `-dataset` is a mandatory parameter specifying the dataset to use. Supported options are `cifar10`, `mnist`, and `cat` (default: `cifar10`).

The script supports the following optional parameters to customize the training phase:
  - `-lr`: the learning rate used during training (default: `0.075`)
  - `-epochs`: the total number of training epochs (default: `80`)


To train the models for the Transfer Learning threat model of this project, I followed these steps:
  1. Loads the chosen dataset and isolates part of the training data to create a validation split, using a fixed random seed (`42`) to ensure reproducibility
  2. Initializes the architecture (`CIFARConvNet` for CIFAR-10, or a standard MLP `NeuralNetwork` for MNIST and Cat-vs-NonCat) and trains it on the clean training set
  3. Tests the newly trained model on the official test split to compute and output the baseline Clean Test Accuracy
  4. Freezes all the hidden layers of the network, leaving only the final classification layer trainable, which perfectly mimics the fixed feature extractor assumption of the Transfer Learning scenario
  5. Saves the frozen model inside the `./models/` directory using the following naming convention: `model_<dataset>_lr<lr>_<epochs>epochs`


# Attack Execution

From the root project directory, the following command can be used:

`python3 poison_attacks.py -file_name <model_file_name> -attack <attack_type>`

The following parameters are mandatory to be specified:
  - `-file_name`: the name of the pre-trained model to be used for the attack
    - can be one of the models in the `models/` directory. The naming pattern is `model_DATASET_lr0.XXX_YYepochs`, where `DATASET` is the dataset used for training, `lr0.XXX` is the learning rate used for training, and `YYepochs` is the number of epochs used for training.
  - `-attack`: the type of attack to be executed. 
    - can be one of the following: `fc`, `polytope`, or `gradient`

The following parameters are optional. Default values are set for binary classification datasets.
  - `-poison_num`: the number of poison samples to generate (default: 10)
  - `-base_class`: the label for the base class (default: 1)
  - `-target_class`: the label for the target class (default: 0)
  - `-epsilon`: the maximum perturbation allowed for the poison samples (default: 0.03)
  - `-watermark_opacity`: the opacity of the watermark to be applied to the poison samples (default: 0.3)
  - `-step_size`: the step size for the optimization algorithm (default: 0.01)
  - `-iterations`: the number of iterations for the optimization algorithm (default: 2000)
  - `-lr`: the learning rate for the optimization algorithm (default: 0.1)
  - `-epochs`: the number of epochs for the optimization algorithm (default: 50)*

Alternatively, it is possible to launch the attack using the `launch.sh` script, which allows to specify the parameters in a more user-friendly way. If this is the chosen way to start the attack, these are the steps:
  1. open the `launch.sh` script from a text editor and modify the parameters at the top of the file to specify the desired attack configuration
  2. launching the script from a linux terminal (or WSL on Windows) using the command: `./launch.sh` from the root directory.

The script will just populate the python command with all the parameters and launch it, printing it on the screen just to allow the user to see what is being executed.


<br>
<br>

# Attack Succes Ratio Evaluation

In order to evaluate the Success Ratio for each attack, I've set up the `evaluate_asr.sh` script that let the user specify all the parameters combination to be tested, and that will launch the attack for each combination, storing all the parameters and results in a csv file, and the outputs of the `poisoning_attack.py` script in a log file. 

This script needs lot of time to be run, especially if we set lot of parameters to be combined and all the models.

I've run this script with for all the 3 models and all the 3 attacks, testing the following parameters combinations:

```sh

# base:target class pairs to be tested for each model
CLASS_PAIRS["model_cat_lr0.0075"]="1:0"
CLASS_PAIRS["model_cifar10_lr0.001_30epochs"]="2:0 5:3 9:1 7:4"
CLASS_PAIRS["model_mnist_lr0.0075"]="1:7 8:3 4:9 6:5"


# How many independent runs per (model, attack, base, target) combo.
REPETITIONS=5

# Fixed attack hyperparameters 
POISON_NUMS=(30 60)
EPSILONS=(0.01 0.03 0.05)
STEP_SIZE=0.01
ITERATIONS=(2000 4000)
WATERMARK_OPACITYS=(0.0 0.2)
LR=0.1
EPOCHS=20
```

With these combinations, the script runs 360 tests for each base:target pair. It took me 3 days to run, but I've collected lots of useful data.

Finally, to evaluate the Attack Success Rate (ASR) there is the `asr_analyze.py` script that takes the results.csv and computes the ASR grouping first by the combination of parameters, and then by model-attack combination.

Here are the final results, that can be found in the `ASR_overall.csv` file. For more detailed results, the `ASR.csv` file can also be used.

| Model | Attack | Runs | Successes | ASR | ASR % |
| --- | :---: | :---: | :---: | :---: | :---: |
| model_cat_lr0.0075 | fc | 120 | 119 | 0.9917 | 99.17% |
| model_cifar10_lr0.001_30epochs | fc | 480 | 73 | 0.1521 | 15.21% |
| model_mnist_lr0.0075 | fc | 480 | 2 | 0.0042 | 0.42% |
| model_cat_lr0.0075 | gradient | 120 | 120 | 1.0 | 100% |
| model_cifar10_lr0.001_30epochs | gradient | 480 | 83 | 0.1729 | 17.29% |
| model_mnist_lr0.0075 | gradient | 480 | 3 | 0.0063 | 0.63% |
| model_cat_lr0.0075 | polytope | 120 | 114 | 0.95 | 95% |
| model_cifar10_lr0.001_30epochs | polytope | 480 | 92 | 0.1917 | 19.17% |
| model_mnist_lr0.0075 | polytope | 480 | 5 | 0.0104 | 1.04% |

---

# Testing Considerations

I've also made up the following guidelines to suggest the choice of the various attack parameters based on the different datasets and attack techniques. These are not strict rules, but they can be used as a starting point for further experimentation or evaluations.

## Base and Target class

For the cat vs non-cat dataset, it is suggested to choose the base class as `1` (non-cat) and the target class as `0` (cat). This is because the cat class is more complex and has more features than the non-cat class, making it easier to generate poisons that can fool the model into misclassifying cats as non-cats. Trying to invert the classes makes every attack fail and the poisoned model to lose accuracy.

Generally, for the MNIST dataset, it is suggested to choose pairs of classes that are visually similar, such as 7 and 1 or 3 and 8.

Some of the best couples are:

| Base | Target |
| --- | :---: |
| 1 | 7 |
| 8 | 3 |
| 4 | 9 |
| 6 | 5 |

On the other hand, for the CIFAR-10 dataset, it is suggested to choose pairs of classes that share similare features in the images, like the background, the shape of the body, etc... There are 10 classes in the CIFAR-10 dataset, here is the official mapping:

| Class | Label |
| --- | :---: |
| airplane | 0 |
| automobile | 1 |
| bird | 2 |
| cat | 3 |
| deer | 4 |
| dog | 5 |
| frog | 6 |
| horse | 7 |
| ship | 8 |
| truck | 9 |

Which brings us with the following pairs of classes that can be used for the attack:

| Base | Target |
| --- | :---: |
| 2 | 0 |
| 9 | 1 |
| 5 | 3 |
| 7 | 4 |

<br>

---


# Some Testing Results

From here on, I will insert some tests and executions i've done before evaluating the ASR (Attack Success Rate) of the attacks, with the parameters used and the results obtained. My goal here was to obtain a baseline for the parameters to start the evaluation process and to show how the different attacks perform under different configurations.

From here on, the results are divided by dataset, and for each dataset, the results are divided by attack type. For each dataset, I tried to train a model and also to find the best parameters for each attack.


# CAT vs NON-CAT Dataset

>[!warning] Disclaimer
> Probably I hadn't the correct and complete dataset, so the models I've tried to train were mostly overfitting. So for the tests on this Dataset, I've used the model provided by the Laboratory of the course: `model_cat_lr0.0075`.
>

## Cat vs Non-Cat Training

I used the following parameters in the train.sh script

`python3 training.py -dataset cat -lr 0.01 -epochs 50`

```txt
Using CUDA device for hardware acceleration
Starting training on Cat-vs-NonCat... (lr=0.01, epochs=50)
Training DNN model with lr 0.01: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:01<00:00, 43.02it/s, Train acc=0.82, Train loss=0.431, Val acc=0.688, Val loss=0.0195]
Baseline Clean Test Accuracy: 0.860
Feature Extractor frozen.
Checkpoint saved with success in: ./model_cat_lr0.01_50epochs
```


## Feature Collision Poisoning Attack on Cat vs Non-Cat

Model:  model_cat_lr0.0075

```txt
Feature Collision Poisoning Attack Implementation

Parameters:
        Base Class: 1;
        Target Class: 0;
        Poison Budget: 30;
        Step Size: 0.01;
        Iterations: 2000;
        Epsilon: 0.03;
        Watermark Opacity: 0.3

        lr: 0.1;
        Epochs: 20

Crafting FC Poisons: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:02<00:00, 738.84it/s, loss=92.1]
        Saved figure to ./images/feature_collision_poisons/cat_clean_images.png
        Saved figure to ./images/feature_collision_poisons/cat_poisoned_images.png
        Saved figure to ./images/feature_collision_poisons/cat_poison_perturbations.png

--------

Results:


Original Prediction: 0.0
Clean Model Accuracy: 0.74

Training DNN model with lr 0.1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 70.40it/s, Train acc=0.952, Train loss=0.137, Val acc=0, Val loss=0]

Poisoned Prediction: 1.0; Success: True
Poisoned Model Accuracy: 0.72
```

## Bullseye Polytope Poisoning Attack on Cat vs Non-Cat

Model:  model_cat_lr0.0075

```txt
Polytope Poisoning Attack Implementation

Parameters:
        Base Class: 1;
        Target Class: 0;
        Poison Budget: 30;
        Step Size: 0.01;
        Iterations: 4000;
        Epsilon: 0.1;
        Watermark Opacity: 0.0

        lr: 0.05;
        Epochs: 25

Crafting Bullseye Polytope Poisons: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4000/4000 [00:06<00:00, 624.03it/s, loss=0.744]
        Saved figure to ./images/polytope_poisons/cat_clean_images_polytope.png
        Saved figure to ./images/polytope_poisons/cat_poisoned_images_polytope.png
        Saved figure to ./images/polytope_poisons/cat_poison_perturbations_polytope.png

--------

Results:


Original Prediction: 0.0
Clean Model Accuracy: 0.74

Training DNN model with lr 0.05: 100%|████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:00<00:00, 73.74it/s, Train acc=0.87, Train loss=0.266, Val acc=0, Val loss=0]

Poisoned Prediction: 1.0; Success: True
Poisoned Model Accuracy: 0.72
```

## Gradient Matching Poisoning Attack on Cat vs Non-Cat

Model:  model_cat_lr0.0075

```txt
Gradient Matching Poisoning Attack Implementation

Parameters:
        Base Class: 1;
        Target Class: 0;
        Poison Budget: 30;
        Step Size: 0.01;
        Iterations: 4000;
        Epsilon: 0.1;

        lr: 0.05;
        Epochs: 25

Crafting Gradient Matching Poisons: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4000/4000 [02:00<00:00, 33.24it/s, cos_sim=0.994]
        Saved figure to ./images/gradient_matching/cat_clean_images_gm.png
        Saved figure to ./images/gradient_matching/cat_poisoned_images_gm.png
        Saved figure to ./images/gradient_matching/cat_poison_perturbations_gm.png

--------

Results:

Original Prediction: 0.0
Clean Model Accuracy: 0.74
Training DNN model with lr 0.05: 100%|██████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:00<00:00, 83.16it/s, Train acc=0.9, Train loss=0.22, Val acc=0, Val loss=0]

Poisoned Prediction: 1.0; Success: True
Poisoned Model Accuracy: 0.72
```


# MNIST Dataset

>[!warning] Disclaimer
> I've pasted here the results of the tests on one of my models, but I used the model provided by the Laboratory of the course for the ASR evaluation, which is `model_mnist_lr0.0075`.
> 
> This choice is due to the fact that the accuracy of the Laboratory model was a bit lower, so it should have been a bit easier to fool the model
> 

The MNIST dataset is a simple and well-defined dataset, which makes it difficult to generate poisons that can fool the model into misclassifying the target instance. This also means that the models converge really fast. For this reasons, every model I've tried to train obtained results that were very similar to the ones obtained with the model provided by the Laboratory of the course: `model_mnist_lr0.0075`. 

## MNIST Training
These are some of the training tests I've run so far:



### First run

`python3 training.py -dataset mnist -lr 0.01 -epochs 20`

```txt
Using CUDA device for hardware acceleration
Starting training on MNIST... (lr=0.01, epochs=20)
Training DNN model with lr 0.01: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:15<00:00,  1.32it/s, Train acc=0.931, Train loss=0.245, Val acc=0.924, Val loss=0.00211]
Baseline Clean Test Accuracy: 0.931
Feature Extractor frozen.
Checkpoint saved with success in: ./model_mnist_lr0.01_20epochs
```

### Second run

`python3 training.py -dataset mnist -lr 0.05 -epochs 20`
        
```txt
Using CUDA device for hardware acceleration
Starting training on MNIST... (lr=0.05, epochs=20)
Training DNN model with lr 0.05: 100%|███████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:13<00:00,  1.50it/s, Train acc=0.982, Train loss=0.0634, Val acc=0.969, Val loss=0.000839]
Baseline Clean Test Accuracy: 0.973
Feature Extractor frozen.
Checkpoint saved with success in: ./model_mnist_lr0.05_20epochs
```

## Feature Collision Poisoning Attack on MNIST

The Feature Collision Attack fails in almonst all MNIST model and base-target combination. MNIST dataset has simple and well-defined features, which makes it difficult to generate poisons that can fool the model into misclassifying the target instance. This is the one of the tests I've run for the MNIST


Model: model_mnist_lr0.05_20epochs 

```txt
Parameters:
        Base Class: 1;
        Target Class: 7;
        Poison Budget: 40;
        Step Size: 0.01;
        Iterations: 2000;
        Epsilon: 0.15;
        Watermark Opacity: 0.15

        lr: 0.01;
        Epochs: 20

--------

Results:

Original Prediction: 7
Clean Model Accuracy: 0.9731

Training DNN model with lr 0.01: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:13<00:00,  1.46it/s, Train acc=0.979, Train loss=0.0838, Val acc=0, Val loss=0]

Poisoned Prediction: 7; Success: False
Poisoned Model Accuracy: 0.9718
```

## Bullseye Polytope Poisoning Attack on MNIST


Model: model_mnist_lr0.05_20epochs 


```txt
Parameters:
        Base Class: 1;
        Target Class: 7;
        Poison Budget: 60;
        Step Size: 0.01;
        Iterations: 2000;
        Epsilon: 0.15;
        Watermark Opacity: 0.15

        lr: 0.01;
        Epochs: 20

--------

Results:

Original Prediction: 7
Clean Model Accuracy: 0.9731

Training DNN model with lr 0.01: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:15<00:00,  1.27it/s, Train acc=0.978, Train loss=0.0848, Val acc=0, Val loss=0]

Poisoned Prediction: 7; Success: False
Poisoned Model Accuracy: 0.9718
```

## Gradient Matching Poisoning Attack on MNIST

Model: model_mnist_lr0.05_20epochs 

```txt
Parameters:
        Base Class: 1;
        Target Class: 7;
        Poison Budget: 60;
        Step Size: 0.005;
        Iterations: 3000;
        Epsilon: 0.1;

        lr: 0.05;
        Epochs: 20

--------

Results:

Original Prediction: 7
Clean Model Accuracy: 0.9731
Training DNN model with lr 0.05: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:14<00:00,  1.41it/s, Train acc=0.981, Train loss=0.0658, Val acc=0, Val loss=0]

Poisoned Prediction: 7; Success: False
Poisoned Model Accuracy: 0.9733
```


# CIFAR Dataset

## Cifar Training

I used the following parameters in the train.sh script

dataset='cifar10'
lr=0.001
epochs=30

The model trained with these parameters is the same used for the ASR evaluation.


## Feature Collision Poisoning Attack on CIFAR-10


Model: model_cifar10_lr0.001_30epochs

```txt
Feature Collision Poisoning Attack Implementation

Parameters:  

    Base Class: 2;  
    Target Class: 0;  
    Poison Budget: 60;  
    Step Size: 0.01;  
    Iterations: 2000;  
    Epsilon: 0.05;  
    Watermark Opacity: 0.2  
    lr: 0.01;
    Epochs: 20

--------

Results:

Original Prediction: 0
Clean Model Accuracy: 0.7641

Training CIFARConvNet model with lr 0.01: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:21<00:00,  1.06s/it, Train acc=0.753, Train loss=0.703, Val acc=0, Val loss=0]

Poisoned Prediction: 2; Success: True
Poisoned Model Accuracy: 0.7315
```

## Polytope Poisoning Attack on CIFAR-10


Model: model_cifar10_lr0.001_30epochs

```txt
Polytope Poisoning Attack Implementation

Parameters:

    Base Class: 2;  
    Target Class: 0;  
    Poison Budget: 60;  
    Step Size: 0.01;  
    Iterations: 2000;  
    Epsilon: 0.04;  
    Watermark Opacity: 0.3  
    lr: 0.01;  
    Epochs: 20  

--------

Results:

Original Prediction: 0
Clean Model Accuracy: 0.7641

Training CIFARConvNet model with lr 0.01: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:19<00:00,  1.01it/s, Train acc=0.754, Train loss=0.703, Val acc=0, Val loss=0]

Poisoned Prediction: 2; Success: True
Poisoned Model Accuracy: 0.7324
```

## Gradient Matching Poisoning Attack on CIFAR-10


Model: model_cifar10_lr0.001_30epochs

```txt
Gradient Matching Poisoning Attack Implementation

Parameters:  

    Base Class: 2;  
    Target Class: 0;  
    Poison Budget: 50;  
    Step Size: 0.01;  
    Iterations: 4000;  
    Epsilon: 0.05;  
    lr: 0.01;  
    Epochs: 20  
 
--------

Results:

Original Prediction: 0
Clean Model Accuracy: 0.7641
Training CIFARConvNet model with lr 0.01: 100%|█████████████████████████| 20/20 [00:26<00:00,  1.34s/it, Train acc=0.754, Train loss=0.699, Val acc=0, Val loss=0]

Poisoned Prediction: 2; Success: True
Poisoned Model Accuracy: 0.7154
```