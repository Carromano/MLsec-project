import math

import matplotlib.pyplot as plt


def plot_images(X, M, N, grayscale=False, title=None):
    M = max(2, math.ceil(len(X) / N))
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M*1.3))
    for i in range(M):
        for j in range(N):
            idx = i*N + j
            if idx >= len(X):
                ax[i][j].set_visible(False)
            else:
                sample = 1 - X[idx].cpu().numpy() if grayscale else X[idx].cpu().numpy()
                ax[i][j].imshow(sample, cmap='gray' if grayscale else None)
                ax[i][j].set_axis_off()
    if title is not None:
        f.suptitle(title)
    plt.tight_layout()
    plt.show()


