# Centered Kernel Alignment (CKA) - PyTorch Implementation

**A PyTorch implementation of Centered Kernel Alignment (CKA) with GPU support.** 

Rather than caching intermediate feature representation, this code calculates CKA on-the-fly (simultaneously with the model forward pass) by using the mini-batch CKA, as described in the [paper by Nguyen et. al.](https://openreview.net/pdf?id=KJNcAkY8tY4)
By leveraging GPU superiority, this implementation runs much faster than any Numpy implementation.

(This code was used for the CKA analysis in our paper. arXiv link coming soon :fire:)

## Setup
I haven't added a `requirements.txt` since the exact version of each package is not that important :man_shrugging:

#### Required packages to use the class/functions:
* python3.7+
* torch (any relatively recent version should be O.K.)
* torchvision 
* tqdm
* torchmetrics

#### To run the `example.ipynb`:
* jupyter
* matplotlib
* numpy

## Example notebook
Try out the example notebook in `example.ipynb`.

## Other
* If you found this repo helpful, please give it a :star:
* If you find any bugs/improvements, please create a new issue.
* This code is mostly tested on ResNets

### TODO (when I feel like it)
* Ditch hooks; change to `torch.fx` implementation