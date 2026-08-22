# MLsec-project
## Clean-Label Poisoning: From Feature Space Constraints to Gradient Matching

This project explores and evaluates the evolution of Clean-Label Data Poisoning attacks against Deep Neural Networks within a Transfer Learning threat model. In this scenario, the attacker cannot alter the labels of the dataset, and the target model acts as a fixed feature extractor while only the final fully connected layer is retrained. The research compares the effectiveness, stealthiness, and transferability of three distinct poisoning methodologies using the `catvnoncat` and `MNIST` datasets.

Methodologies Evaluated:  
  1. **Feature Collision:** A direct approach that generates poisoned samples by minimizing the $L_2$ Euclidean distance between the poison and the target instance within the feature space, enforced by an $L_\infty$ perturbation bound to maintain visual stealthiness.
  2. **Convex Polytope:** A relaxed optimization strategy that generates a multi-poison convex hull around the target instance ($\sum_{j=1}^k c_j \phi(x_p^j)$). By fixing the distribution coefficients to $1/k$ (Bullseye Polytope), this approach centers the target within the polytope, reducing computational overhead and significantly improving transferability.
  3. **Gradient Matching:** An advanced technique that shifts the focus from the feature space to the learning dynamics. The poisons are optimized by minimizing the negative cosine similarity of their loss gradients relative to the target instance's gradients ($\nabla_\theta \mathcal{L}$), hijacking the gradient descent direction during training.

All the methodologies are finally evaluated against a third model, trained by myself on the `CIFAR-10` dataset.

## Cifar-10 training

I've written the train_cifar10.py script to train a model on the CIFAR-10 dataset and freeze it for later use in poisoning attacks. The script loads the CIFAR-10 dataset from torchvision, defines a neural network architecture from the one used in the last lab of the course, trains the model on the dataset, and saves it to a file which respected the naming convention expected by the poison_attacks.py script. The model is trained for 50 epochs with a learning rate of 0.0075, and the final accuracy on the test set is printed to the console. The model is then saved to a file named "model_cifar10_lr0.0075".


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

---

<br>


## Project structure

```text
MLsec-project/
├── poison_attacks.py                   # Poison attack routines and experiments.
│
├── poisoning/                          # Dataset loading and poison-crafting modules.
│   ├── constants.py                     # Shared constants used by the scripts.
│   ├── dataset.py                       # Dataset loading, preprocessing, and preparation.
│   ├── neural_network.py                # Neural network definition and training.
│   ├── plotting.py                      # Plotting and visualization utilities.
│   └── poison_crafting.py               # Poison data generation for the experiments.
│
├── data/                               # Dataset files for the experiments.
│   ├── test_catvnoncat.h5               # Test dataset for the cat-vs-non-cat task.
│   ├── train_catvnoncat.h5              # Training dataset for the cat-vs-non-cat task.
│   └── MNIST/                           # MNIST dataset, divided into train/test splits.
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
├── models/                             # Pre-trained model artifacts.
│   ├── model_cat_lr0.0075               # Pre-trained model for cat-vs-non-cat.
│   └── model_mnist_lr0.0075             # Pre-trained model for MNIST.
│
├── requirements.txt                     # Python dependencies for the project.
├── README.md                            # Project documentation.
└── LICENSE                              # GitHub repository license.
```

---

<br>
<br>

# Attack Execution

from the root project directory, the following command can be used: 

`python3 poison_attacks.py`

the following parameters are mandatory to be specified:
- `-file_name`: the name of the pre-trained model to be used for the attack
  - can be one of the following: `model_cat_lr0.0075` or `model_mnist_lr0.0075`
  - the choice of the model will determine the dataset to be used for the attack
- `-attack`: the type of attack to be executed. 
  - can be one of the following: `fc`, `polytope`, or `gradient`

The following parameters are optional and can be used to specify the base class and target class for the attack. Default values are set for binary classification datasets.
- `-poisons`: the number of poison samples to generate (default: 10)
- `-base_class`: the label for the base class (default: 1)
- `-target_class`: the label for the target class (default: 0)
- `-epsilon`: the maximum perturbation allowed for the poison samples (default: 0.03)
- `-watermark_opacity`: the opacity of the watermark to be applied to the poison samples (default: 0.3)
- `-step_size`: the step size for the optimization algorithm (default: 0.01)
- `-iterations`: the number of iterations for the optimization algorithm (default: 2000)
- `-lr`: the learning rate for the optimization algorithm (default: 0.1)
- `-epochs`: the number of epochs for the optimization algorithm (default: 50)

**Alternatively**, is it possible to launch the attack using the `launch.sh` script, which allows to specify the parameters in a more user-friendly way. If this is the chosen way to launche the attack, these are the steps:
  1. open the `launch.sh` script and edit the parameters at the top of the file to specify the desired attack configuration
  2. launching the script from a linux terminal (or WSL on Windows) using the command: `./launch.sh` from the root directory.

The script will just populate the python command with all the parameters and launch it, printing it on the screen just to allow the user to see what is being executed.

---

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


---

## Feature Collision Attack

### Cat vs Non-Cat parameters
| Parameter | Suggested Value |
| --- | :---: |
| poison budget | 30-50 |
| epsilon | 0.03 - 0.06 | 
| watermark_opacity | 0.2 - 0.3 |
| Step Size | 0.01 |
| iterations | 2000 |

### MNIST parameters
| Parameter | Suggested Value |
| --- | :---: |
| poison budget | 10-30 |
| epsilon | 0.10 - 0.20 | 
| watermark_opacity | 0.1 - 0.2 |
| step size | 0.01 - 0.05 | 
| iterations | 1000 | 

### CIFAR-10 parameters
| Parameter | Suggested Value |
| --- | :---: |
| poison budget | 40-60 |
| epsilon | 0.03 - 0.06 |
| watermark_opacity | 0.2 - 0.3 |
| Step Size | 0.01 |
| iterations | 2000 |

---

## Convex Polytope Attack

### Cat vs Non-Cat parameters
| Parameter | Suggested Value |
| --- | :---: |
| poison budget | 50-100 |
| epsilon | 0.04- 0.08 |
| watermark_opacity | 0.2 - 0.3 |
| Step Size | 0.005 - 0.01 |
| iterations | 3000-4000 |

### MNIST parameters
| Parameter | Suggested Value |
| --- | :---: |
| poison budget | 30-50 |
| epsilon | 0.10 - 0.30 |
| watermark_opacity | 0.0 - 0.1 |
| Step Size | 0.01 |
| iterations | 2000 |

### CIFAR-10 parameters
| Parameter | Suggested Value |
| --- | :---: |
| poison budget | 50-100 |
| epsilon | 0.04 - 0.08 |
| watermark_opacity | 0.0 - 0.2 |
| Step Size | 0.005 - 0.01 |
| iterations | 3000-4000 |

---

## Gradient Matching Attack

### Cat vs Non-Cat parameters
| Parameter | Suggested Value |
| --- | :---: |
| poison budget | 40-80 |
| epsilon | 0.05 - 0.10 |
| watermark_opacity | 0.0 |
| Step Size | 0.01 - 0.05 |
| iterations | 4000-5000 |

### MNIST parameters
| Parameter | Suggested Value |
| --- | :---: |
| poison budget | 20-40 |
| epsilon | 0.15 - 0.30 |
| watermark_opacity | 0.0 |
| Step Size | 0.05 |
| iterations | 2000-3000 |

### CIFAR-10 parameters
| Parameter | Suggested Value |
| --- | :---: |
| poison budget | 50-100 |
| epsilon | 0.05 - 0.10 |
| watermark_opacity | 0.0 |
| Step Size | 0.01 - 0.05 |
| iterations | 4000-5000 |
