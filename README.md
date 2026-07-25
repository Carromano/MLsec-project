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
├── poison_attacks.py                       # Poison attack routines and experiments.
│
├── poisoning/                                   # Dataset directory containing MNIST and cat-vs-non-cat splits.
|   ├── constants.py                            # Shared constants used by the scripts.
|   ├── dataset.py                              # Dataset loading, preprocessing, and preparation.
|   ├── neural_network.py                       # Neural network definition and training.
|   ├── plotting.py                             # Plotting and visualization utilities.
|   └── poison_crafting.py                      # Poison data generation for the experiments.
│
│
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


## Esecuzione attacco FC poisoning

dalla cartella principale del progetto, eseguire il seguente comando per lanciare l'attacco FC poisoning:

- su dataset cat-vs-non-cat:
  `python3 poison_attacks.py -file_name model_cat_lr0.0075`
- su dataset MNIST:
  `python3 poison_attacks.py -file_name model_mnist_lr0.0075`


### Primo risultato

$ python3 poison_attacks.py -file_name model_cat_lr0.0075

Using CUDA device for hardware acceleration
Crafting FC Poisons: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:06<00:00, 310.66it/s, loss=0.77]
plotting.py:21: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
Original Prediction: 0.0
Clean Model Accuracy: 0.74
Training DNN model with lr 0.1: 100%|██████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 51.82it/s, Train acc=0.943, Train loss=0.108, Val acc=0, Val loss=0]
Poisoned Prediction: 1.0; Success: True
Poisoned Model Accuracy: 0.72


$ python3 poison_attacks.py -file_name model_mnist_lr0.0075
Using CUDA device for hardware acceleration
Crafting FC Poisons: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:07<00:00, 268.68it/s, loss=22.7]
/mnt/c/Users/romat/OneDrive - uniroma1.it/Magistrale/2° Anno/2o Semestre/Machine Learning Security/Progetto/MLsec-project/poisoning/plotting.py:21: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
Original Prediction: 0
Clean Model Accuracy: 0.9266
Training DNN model with lr 0.1: 100%|███████████████████████████████████████████████████████████████████████████████████| 50/50 [01:29<00:00,  1.79s/it, Train acc=0.93, Train loss=0.252, Val acc=0, Val loss=0]
Poisoned Prediction: 0; Success: False
Poisoned Model Accuracy: 0.9196




