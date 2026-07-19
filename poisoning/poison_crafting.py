import tqdm
import torch


def craft_fc_poisons(extractor, base_imgs, target_img, step_size, iterations=1000, epsilon=0.03, watermark_opacity=0.3):
    """ computes feature collision poisons and returns perturbation delta for each base image """
    extractor.eval()

    # initialize poisons and add low-opacity watermark
    x_poisons = (1 - watermark_opacity) * base_imgs + watermark_opacity * target_img
    x_poisons = x_poisons.clone().detach().requires_grad_(True)
                                                          

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
        # L2 distance: ||f(p) - f(t)||^2
        loss = torch.nn.functional.mse_loss(poison_features, target_features, reduction='none').mean(dim=1)

        # compute gradients of loss wrt poisoned sample
        grads = torch.autograd.grad(loss.sum(), x_poisons)[0]

        with torch.no_grad():
            # optimize poisons
            x_poisons -= step_size * grads.sign() # Uso del segno per robustezza (FGSM-like)

            # constrain within allowed eps-perturbation and [0, 1] domain
            delta = torch.clamp(x_poisons - base_imgs, min=-epsilon, max=epsilon)
            x_poisons.data = torch.clamp(base_imgs + delta, min=0.0, max=1.0)

            # add logging of loss to progress bar during optimization
            progress_bar.set_postfix({'loss': loss.mean().item()})

    # return poison perturbation (not full poisoned sample)
    return (x_poisons.detach() - base_imgs).cpu()
