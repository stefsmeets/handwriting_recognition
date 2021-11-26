import numpy as np
import matplotlib.pyplot as plt

def display_data(X, patch_width=None):
    """This function displays the training samples in 'X' on a grid."""
    m, n = X.shape
    patch_width = None

    if not patch_width:
        patch_width = int(np.round(np.sqrt(X.shape[1])))

    patch_height = int((n / patch_width))

    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    pad = 1

    display_array = np.ones((pad + display_rows * (patch_height + pad), 
                            pad + display_cols * (patch_width + pad)))

    index = 0

    for i in range(display_rows):
        for j in range(display_cols):
            j0 = pad + j * (patch_height + pad)
            j1 = j0 + patch_height
            
            i0 = pad + i * (patch_width + pad)
            i1 = i0 + patch_width
            
            patch = X[index].reshape(patch_height, patch_width)
            max_val = np.abs(patch).max()
            
            display_array[j0:j1, i0:i1] = (patch / max_val).T

            index += 1

    plt.figure()
    plt.imshow(display_array, cmap='gray')