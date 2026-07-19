# Feature Collision Project Report

## `constants.py`
- **Purpose:** Central configuration for dataset and model paths plus a small numerical constant.
- **Functions and classes exposed:** No functions or classes; only module-level constants: `data_dir`, `model_dir`, `batch_size`, and `epsilon`.
- **PyTorch mapping:** None directly.
- **Feature Collision relevance:** No attack logic; this file only supplies shared paths and defaults.

## `dataset.py`
- **Purpose:** Loads MNIST or cat-vs-non-cat data, normalizes it, flattens images, and wraps train/test tensors in loaders.
- **Functions and classes exposed:** `flatten()`, `compute_mean_std()`, `load_cat_noncat()`, `mnist_unpack()`, `load_mnist()`, `get_dataset()`, and the `Dataset` class with methods `is_grayscale()`, `unnormalize_data()`, `get_sub_testloader()`, `get_test_data()`, `get_train_data()`, `poison_data()`, `get_unnormalized_testset()`, and `unflatten()`.
- **PyTorch mapping:** `torch.from_numpy()`, `torch.std_mean()`, `TensorDataset`, and `DataLoader` are used directly. The key pixel scaling happens in `get_dataset()`, where raw image values are divided by `255.` to map inputs into `[0, 1]`. `Dataset.unnormalize_data()` reverses that mapping for visualization.
- **Feature Collision relevance:** `Dataset.poison_data()` is the mutation point where the crafted delta is applied to selected base samples.

## `neural_network.py`
- **Purpose:** Defines the classifier, exposes a feature extractor interface, and implements training, evaluation, and serialization.
- **Functions and classes exposed:** `get_torch_device()` and the `NeuralNetwork` class with methods `__init__()`, `forward()`, `features()`, `from_pretrained()`, `_pred()`, `predict()`, `freeze_extractor()`, `fit()`, `test()`, `save()`, and `load()`.
- **PyTorch mapping:** Uses `nn.Linear`, `nn.Sequential`, activation modules, `loss.backward()`, `torch.no_grad()`, `torch.save()`, and `torch.load()`. The model is explicitly structured so `features()` returns the embedding before the final classifier layer.
- **Feature Collision relevance:** `features()` is the key hook for the attack objective, since FC optimizes in feature space rather than directly on logits.

## `plotting.py`
- **Purpose:** Visualizes clean and poisoned images in grid form.
- **Functions and classes exposed:** `plot_images()` only.
- **PyTorch mapping:** Uses `.cpu().numpy()` to move tensors to host memory before plotting.
- **Feature Collision relevance:** Diagnostic only; used to inspect poison appearance after optimization.

## `poison_attacks.py`
- **Purpose:** Orchestrates the full clean-label Feature Collision experiment.
- **Functions and classes exposed:** `get_images_from_label()`, `main()`, and `get_data_and_model()`.
- **PyTorch mapping:** Most tensor operations are delegated to `dataset.py`, `neural_network.py`, and `poison_crafting.py`. This file mainly coordinates the attack pipeline through those PyTorch-backed APIs.
- **Feature Collision relevance:** The driver for the attack experiment. Variables such as `step_size`, `iterations`, `epsilon`, `watermark_opacity`, `base_class`, `target_class`, `poison_num`, `base_imgs`, `target_img`, and `delta` define the attack setup.

## `poison_crafting.py`
- **Purpose:** Intended to implement the core poison-generation loop for Feature Collision.
- **Functions and classes exposed:** `craft_fc_poisons()` only.
- **PyTorch mapping:** The file currently contains only minimal PyTorch usage: `extractor.eval()` and a `torch.no_grad()` block reserved for the update step. There is no active `torch.autograd.grad()` call and no `torch.clamp()`-style projection in the current implementation.
- **Feature Collision relevance:** This is the missing core of the attack. The intended objective is to minimize the $L_2$ distance between poisoned and target samples in feature space while keeping perturbations inside an $L_\infty$ bound, but that logic is not yet implemented here.

## Overall status
- The repository already contains the dataset handling, model feature-extraction interface, and attack orchestration.
- The actual FC optimization loop is not implemented yet in `poison_crafting.py`; it currently serves as a scaffold for the poison-update objective and the $L_\infty$ projection step.