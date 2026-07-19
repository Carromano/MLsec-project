# MLsec-project
## Clean-Label Poisoning: From Feature Space Constraints to Gradient Matching

This project explores and evaluates the evolution of Clean-Label Data Poisoning attacks against Deep Neural Networks within a Transfer Learning threat model. In this scenario, the attacker cannot alter the labels of the dataset, and the target model acts as a fixed feature extractor while only the final fully connected layer is retrained. The research compares the effectiveness, stealthiness, and transferability of three distinct poisoning methodologies using the `catvnoncat` and `MNIST` datasets.

Methodologies Evaluated:  
  1. **Feature Collision:** A direct approach that generates poisoned samples by minimizing the $L_2$ Euclidean distance between the poison and the target instance within the feature space, enforced by an $L_\infty$ perturbation bound to maintain visual stealthiness.
  2. **Convex Polytope:** A relaxed optimization strategy that generates a multi-poison convex hull around the target instance ($\sum_{j=1}^k c_j \phi(x_p^j)$). By fixing the distribution coefficients to $1/k$ (Bullseye Polytope), this approach centers the target within the polytope, reducing computational overhead and significantly improving transferability.
  3. **Gradient Matching:** An advanced technique that shifts the focus from the feature space to the learning dynamics. The poisons are optimized by minimizing the negative cosine similarity of their loss gradients relative to the target instance's gradients ($\nabla_\theta \mathcal{L}$), hijacking the gradient descent direction during training.

---

## Requirements

### Python
- Python ≥ 3.9

### External libraries
```
pip install torch torchvision torchaudio numpy matplotlib
```

---

## Project structure

```text
MLsec-project/
├── constants.py                            # Shared constants used by the scripts.
├── dataset.py                              # Dataset loading, preprocessing, and preparation.
├── neural_network.py                       # Neural network definition and training.
├── plotting.py                             # Plotting and visualization utilities.
├── poison_attacks.py                       # Poison attack routines and experiments.
├── poison_crafting.py                      # Poison data generation for the experiments.
│
├── data/                                   # Dataset directory containing MNIST and cat-vs-non-cat splits.
│   ├── test_catvnoncat.h5                      # Test dataset for the cat vs non-cat task.
│   ├── train_catvnoncat.h5                     # Training dataset for the cat vs non-cat task.
│   └── MNIST/                                  # MNIST dataset directory, divided into train/test splits.
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
├── models/                                 # pre-trained model artifacts for the experiments.
|   ├── model_cat_lr0.0075                      # pre-trained model on the catvsnoncat dataset.
|   └── model_mnist_lr0.0075                    # pre-trained model on the MNIST dataset.
|
├── requirements.txt                        # Python dependencies for the project.
├── README.md                               # Project documentation
└── LICENSE                                 # GitHub repository license.
```

---