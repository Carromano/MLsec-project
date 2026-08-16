import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def plot_images(X, M, N, grayscale=False, title=None, filename=None, isPerturbation=False):

    # determine the number of rows needed to display all images
    M = max(2, math.ceil(len(X) / N))
    
    # create a figure with M rows and N columns of subplots, sharing x and y axes, and set the figure size
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M*1.3))
    
    # flatten the axes array if M or N is 1 to simplify indexing
    for i in range(M):
        for j in range(N):
            idx = i*N + j

            # if the index exceeds the number of images, hide the subplot
            if idx >= len(X):
                ax[i][j].set_visible(False)
            # else, display the image in the subplot
            else:
                sample = 1 - X[idx].cpu().numpy() if grayscale else X[idx].cpu().numpy()

                # to print correctly also the perturbations of RGB images, we need to normalize them to [0, 1] range
                if isPerturbation:
                    sample = np.clip(sample, 0.0, 1.0)
                    if not grayscale and sample.ndim == 3 and sample.shape[0] == 3:
                        sample = np.transpose(sample, (1, 2, 0))

                ax[i][j].imshow(sample, cmap='gray' if grayscale else None)
                ax[i][j].set_axis_off()

    # set the title of the figure if provided
    if title is not None:
        f.suptitle(title)

    # adjust layout to prevent overlap of subplots and title
    plt.tight_layout()

    # save the figure if filename is provided, otherwise show it
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"\tSaved figure to {filename}")
        
    # show the figure if the backend if it is interactive (gave error before on WSL)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()

    # close the figure to free memory
    plt.close(f)



