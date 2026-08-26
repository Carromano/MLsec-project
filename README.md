# MLsec-project
## Clean-Label Poisoning: From Feature Space Constraints to Gradient Matching

This project explores and evaluates the evolution of Clean-Label Data Poisoning attacks against Neural Networks within a Transfer Learning threat model. In this scenario, the attacker cannot alter the labels of the dataset, and the target model acts as a fixed feature extractor while only the final layer is retrained. The research compares the effectiveness, stealthiness, and transferability of three distinct poisoning methodologies using the `catvnoncat`, `MNIST` and `CIFAR-10` datasets. The final goal is to try to mislead the model into misclassifying a specific target instance by injecting a small number of crafted poison samples into the training dataset.

Methodologies Evaluated:  
  1. **Feature Collision:** A targeted approach that minimizes the $L_2$ Euclidean distance between the poisoned sample and the target instance within the model's feature space. To ensure visual stealthiness, the perturbation is strictly bounded by an $L_\infty$ norm applied directly in the input space.
  2. **Convex Polytope (Bullseye):** A multi-poison strategy that traps the target instance inside a convex hull formed by the poisons in the feature space. By assigning equal weights ($1/k$) to the distribution coefficients, the target is perfectly centered (Bullseye). This structural relaxation drastically improves the attack's transferability to unknown black-box models.
  3. **Gradient Matching:** An optimization technique that shifts the focus from feature space representations to the learning dynamics. It generates poisons by minimizing the negative cosine similarity between their loss gradients and the target's gradients ($\nabla_\theta \mathcal{L}$). This alignment effectively hijacks the gradient descent direction during the final layer retraining.


---

## Requirements

### Python
- Python ≥ 3.9

### External libraries
```sh
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
│   ├── cifar-10-python.tar.gz         # CIFAR-10 archive (downloaded/used for training) - not present on the repository due to size constraints
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
│   ├── feature_collision_poisons/           # Feature Collision attack visualizations.
│   ├── gradient_matching/                   # Gradient Matching attack visualizations.
│   └── polytope_poisons/                    # Bullseye Polytope attack visualizations.
|
├── models/                            # Pre-trained and frozen model artifacts
│   └── model_DATASET_lr0.XXX_YYepochs       # Naming pattern for pre-trained models: Dataset name, learning rate, and number of epochs used for training.
|                                            # The only 2 exceptions are the ones taken from the Laboratory of the course, which are named model_cat_lr0.0075 and model_mnist_lr0.0075.
│
├── requirements.txt                   # Python dependencies for the project.
├── README.md                          # Project documentation and report
├── LICENSE                            # GitHub repository license.
└── .gitignore                         # Repository ignore rules.
```

<br>
<br>


# Training the models

*From the root project directory, the baseline models can be trained and prepared using the following command:

`python3 training.py -dataset <dataset_name>`

The script supports the following optional parameters to customize the training phase:
- `-dataset`: the dataset to use. Supported options are `cifar10`, `mnist`, and `cat` (default: `cifar10`).
- `-lr`: the learning rate used during training (default: `0.075`)
- `-epochs`: the total number of training epochs (default: `80`)


To try to replicate the Transfer Learning threat model evaluated in this project, the script automatically executes a 5-step pipeline
1. Loads the chosen dataset and isolates 20% of the training data to create a validation split, using a fixed random seed (`42`) to ensure reproducibility
2. Initializes the architecture (`CIFARConvNet` for CIFAR-10, or a standard MLP `NeuralNetwork` for MNIST and Cat-vs-NonCat) and trains it on the clean training set
3. Tests the newly trained model on the official test split to compute and output the baseline Clean Test Accuracy
4. Freezes all the hidden layers of the network (the backbone). This leaves only the final classification layer trainable, which perfectly mimics the fixed feature extractor assumption of the Transfer Learning scenario
5. Saves the frozen model inside the `./models/` directory using the specific naming convention `model_<dataset>_lr<lr>_<epochs>epochs`, making it ready to be targeted by the clean-label poisoning attacks


# Attack Execution

*From the root project directory, the following command can be used:

`python3 poison_attacks.py -file_name <model_file_name> -attack <attack_type>`

the following parameters are mandatory to be specified:
- `-file_name`: the name of the pre-trained model to be used for the attack
  - can be one of the models in the `models/` directory. The naming pattern is `model_DATASET_lr0.XXX_YYepochs`, where `DATASET` is the dataset used for training, `lr0.XXX` is the learning rate used for training, and `YYepochs` is the number of epochs used for training.
- `-attack`: the type of attack to be executed. 
  - can be one of the following: `fc`, `polytope`, or `gradient`

The following parameters are optional and can be used to specify the base class and target class for the attack. Default values are set for binary classification datasets.
- `-poison_num`: the number of poison samples to generate (default: 10)
- `-base_class`: the label for the base class (default: 1)
- `-target_class`: the label for the target class (default: 0)
- `-epsilon`: the maximum perturbation allowed for the poison samples (default: 0.03)
- `-watermark_opacity`: the opacity of the watermark to be applied to the poison samples (default: 0.3)
- `-step_size`: the step size for the optimization algorithm (default: 0.01)
- `-iterations`: the number of iterations for the optimization algorithm (default: 2000)
- `-lr`: the learning rate for the optimization algorithm (default: 0.1)
- `-epochs`: the number of epochs for the optimization algorithm (default: 50)*

Alternatively, is it possible to launch the attack using the `launch.sh` script, which allows to specify the parameters in a more user-friendly way. If this is the chosen way to start the attack, these are the steps:
  1. open the `launch.sh` script from a text editor and modify the parameters at the top of the file to specify the desired attack configuration
  2. launching the script from a linux terminal (or WSL on Windows) using the command: `./launch.sh` from the root directory.

The script will just populate the python command with all the parameters and launch it, printing it on the screen just to allow the user to see what is being executed.


<br>
<br>

# Considerations
After some testing, I've made up the following guidelines to suggest the choice of the various attack parameters based on the different datasets and attack techniques. These are not strict rules, but they can be used as a starting point for further experimentation.


## Base and Target class

For the cat vs non-cat dataset, it is suggested to choose the base class as `1` (non-cat) and the target class as `0` (cat). This is because the cat class is more complex and has more features than the non-cat class, making it easier to generate poisons that can fool the model into misclassifying cats as non-cats. Trying to invert the classes makes every attack fail and the poisoned model to lose accuracy.

Generally, for the MNIST dataset, it is suggested to choose pairs of classes that are visually similar, such as 7 and 1 or 3 and 8.

Some of the best couples are:

| Target | Base |
| --- | :---: |
| 7 | 1 |
| 3 | 8 |
| 9 | 4 |
| 5 | 6 |


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

| Target | Base |
| --- | :---: |
| 0 | 2 |
| 1 | 9 |
| 3 | 5 |
| 4 | 7 |

<br>

# Some parameters suggestion and results

## CIFAR


## CIFAR
model_file='model_cifar10_lr0.0075_80epochs'

## ATTACKS
attack_type='fc'
# attack_type='polytope'
# attack_type='gradient'



# Examples of Executions

## Cifar Training

I used the following parameters in the train.sh script

dataset='cifar10'
lr=0.001
epochs=30


## Feature Collision Poisoning Attack on CIFAR-10

Model: model_cifar10_lr0.001_30epochs

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

## Polytope Poisoning Attack on CIFAR-10

Model: model_cifar10_lr0.001_30epochs

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


## Gradient Matching Poisoning Attack on CIFAR-10

Model: model_cifar10_lr0.001_30epochs

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