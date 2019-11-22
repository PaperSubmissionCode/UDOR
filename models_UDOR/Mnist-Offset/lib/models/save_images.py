"""
Image grid saver, based on color_grid_vis from github.com/Newmu
and https://github.com/igul222/improved_wgan_training/blob/master/tflib/save_images.py
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave

def save_images(X, save_path, n_rows=None, n_cols=None):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')
    #print("X in function")
    #print(X)

    n_samples = X.shape[0] # batch_size，也就是多少张小图
    
    if n_rows is None:        
        n_rows = int(np.sqrt(n_samples)) # n_samples=64=batch_size
        while n_samples % n_rows != 0:
            n_rows -= 1
    if n_cols is None:
        n_cols = n_samples // n_rows
    
    nh, nw = n_rows, n_cols
    '''
    print("Xndim")
    print(X.ndim)
    print(nh)
    print(nw)
    '''
    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1) # todo：[B,C,H,W] to [B,H,W,C]
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n//nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x
    
    #print(img)
        #print(x[0])

    imsave(save_path, img)