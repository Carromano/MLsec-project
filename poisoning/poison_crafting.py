import tqdm
import torch


# computes feature collision poisons and returns perturbation delta for each base image
def craft_fc_poisons(extractor, base_imgs, target_img, step_size, iterations=1000, epsilon=0.03, watermark_opacity=0.3):
    extractor.eval()

    # initialize poisons and add low-opacity watermark
    with torch.no_grad():
        watermarked_images = (1 - watermark_opacity) * base_imgs + watermark_opacity * target_img
        watermarked_images = torch.clamp(watermarked_images, min=0.0, max=1.0)

    x_poisons = watermarked_images.clone().detach().requires_grad_(True)

    # extract features of target image
    with torch.no_grad():
        target_features = extractor.features(target_img)

    # duplicate it as many times as base images to later compute loss
    target_features = target_features.repeat(base_imgs.size(0), 1)

    progress_bar = tqdm.trange(iterations, desc='Crafting FC Poisons')

    # iterative optimization. If you want to optimize many poisons, you can also split the optimization in minibatches
    for _ in progress_bar:
        # extract features of poison data
        poison_features = extractor.features(x_poisons)

        # compute individual loss for each poisoned datapoint
        #   L2 distance: ||f(p) - f(t)||^2
        loss = torch.nn.functional.mse_loss(poison_features, target_features, reduction='sum') 

        # compute gradients of loss wrt poisoned sample
        grads = torch.autograd.grad(loss, x_poisons)[0]

        with torch.no_grad():
            # optimize poisons
            x_poisons -= step_size * grads.sign() 

            # constrain within allowed eps-perturbation and [0, 1] domain
            delta = torch.clamp(x_poisons - base_imgs, min=-epsilon, max=epsilon)
            x_poisons.data = torch.clamp(base_imgs + delta, min=0.0, max=1.0)

            # add logging of loss to progress bar during optimization
            progress_bar.set_postfix({'loss': loss.mean().item()})

    # return poison perturbation (not full poisoned sample)
    return (x_poisons.detach() - base_imgs).cpu()



# Computes convex polytope (bullseye) poisons and returns the perturbation delta for each base image 
def craft_polytope_poisons(extractor, base_imgs, target_img, step_size, iterations=1000, epsilon=0.03, watermark_opacity=0.3):
    extractor.eval()

    # initialize poisons and add low-opacity watermark
    with torch.no_grad():
        watermarked_images = (1 - watermark_opacity) * base_imgs + watermark_opacity * target_img
        watermarked_images = torch.clamp(watermarked_images, min=0.0, max=1.0)

    x_poisons = watermarked_images.clone().detach().requires_grad_(True)

    # extract features of target image
    with torch.no_grad():
        target_features = extractor.features(target_img)    # Shape: (1, feature_dim)

    progress_bar = tqdm.trange(iterations, desc='Crafting Bullseye Polytope Poisons')

    # optimization loop for crafting poisons
    for _ in progress_bar:

        # Extract features of the current poison batch
        poison_features = extractor.features(x_poisons)     # Shape: (num_poisons, feature_dim)

        # Compute the centroid (mean) of the poison features in the batch
        # This implements the 1/k sum constraint for the polytope center (Bullseye strategy)
        mean_poison_features = poison_features.mean(dim=0, keepdim=True)

        # Compute MSE loss between the mean poison feature vector and the target feature vector
        # || phi(t) - (1/k) * sum(phi(p_j)) ||^2
        loss = torch.dist(mean_poison_features, target_features, p=2)

        # Compute gradients of the loss with respect to all poisoned samples
        grads = torch.autograd.grad(loss, x_poisons)[0]

        with torch.no_grad():
            # Optimize poisons using sign-based gradient descent (FGSM-like robust step)
            x_poisons -= step_size * grads.sign()

            # Constrain within allowed L-infinity epsilon-perturbation and valid [0, 1] pixel domain
            delta = torch.clamp(x_poisons - base_imgs, min=-epsilon, max=epsilon)

            # Update the poisoned samples to be the base images plus the constrained perturbation
            x_poisons.data = torch.clamp(base_imgs + delta, min=0.0, max=1.0)

            # Logging loss to the progress bar
            progress_bar.set_postfix({'loss': loss.item()})

    # Return the final perturbation delta (not the full poisoned sample) transferred to CPU
    return (x_poisons.detach() - base_imgs).cpu()



# Computes gradient matching poisons and returns the perturbation delta for each base image 
def craft_gradient_matching_poisons(extractor, base_imgs, target_img, step_size, iterations=1000, epsilon=0.03):
    extractor.eval()

    # Inizializziamo partendo dalle immagini base pulite
    x_poisons = base_imgs.clone().detach().requires_grad_(True)
    t_img = target_img if target_img.dim() == 4 else target_img.unsqueeze(0)

    # extracting target features
    target_features = extractor.features(t_img)
    
    # computing all the parameters' gradients with respect to the target features
    all_params = list(extractor.parameters())
    all_grads = torch.autograd.grad(target_features.sum(), all_params, allow_unused=True)
    
    # obtaining only valid parameters and gradients (some parameters may not contribute to the target features)
    valid_params = []
    valid_grads = []
    for p, g in zip(all_params, all_grads):
        if g is not None:
            valid_params.append(p)
            valid_grads.append(g)
    
    # selecting only the last two layers' parameters and gradients for gradient matching
    match_params = valid_params[-2:]
    target_grads = valid_grads[-2:]

    # flattening the target gradients and repeating them for each base image to compute the loss
    target_grads_flat = torch.cat([g.reshape(-1) for g in target_grads]).unsqueeze(0).detach()
    target_grads_flat = target_grads_flat.repeat(base_imgs.size(0), 1)

    # iterative optimization
    progress_bar = tqdm.trange(iterations, desc='Crafting Gradient Matching Poisons')

    for _ in progress_bar:
        # array to hold flattened gradients for each poison sample
        poison_grads_list = []

        # compute gradients of poison features wrt poisoned samples for each poison in the batch
        for i in range(x_poisons.size(0)):
            # extract features of the current poison sample
            p_feat = extractor.features(x_poisons[i:i+1])
            
            # compute gradients of poison features wrt model parameters for the current poison sample
            p_grads = torch.autograd.grad(p_feat.sum(), match_params, create_graph=True)
            p_grads_flat = torch.cat([g.reshape(-1) for g in p_grads])
            poison_grads_list.append(p_grads_flat)

        # stack the list of flattened poison gradients into a single tensor for loss computation
        poison_grads_flat = torch.stack(poison_grads_list)

        # compute cosine similarity between poison gradients and target gradients
        cos_sim = torch.nn.functional.cosine_similarity(poison_grads_flat, target_grads_flat, dim=1)

        # compute loss as 1 - cosine similarity (we want to maximize similarity, so minimize this loss)
        # sum over all poisoned samples to get a single scalar loss value
        loss = (1.0 - cos_sim).sum()

        # compute gradients of loss wrt poisoned samples
        grads = torch.autograd.grad(loss, x_poisons)[0]

        with torch.no_grad():
            # optimize poisons using sign-based gradient descent (FGSM-like)
            x_poisons -= step_size * grads.sign()

            # constrain within allowed eps-perturbation and [0, 1] domain
            delta = torch.clamp(x_poisons - base_imgs, min=-epsilon, max=epsilon)
            x_poisons.data = torch.clamp(base_imgs + delta, min=0.0, max=1.0)

            # add logging of loss to progress bar during optimization
            progress_bar.set_postfix({'cos_sim': cos_sim.mean().item()})

    # return poison perturbation (not full poisoned sample)
    return (x_poisons.detach() - base_imgs).cpu()
