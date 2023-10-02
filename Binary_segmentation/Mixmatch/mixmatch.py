import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import utils

def Data_Augmentation(Imgs, masks):
    """ augmentation for a batch of images
    Args:
        Imgs : Data,     NxCxHxW
        mask : label Nx1xHxW
    Return: augmentated images and masks,  NxCxHxW
    """
    transform = transforms.Compose([transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    X = transform(Imgs)
    Y = masks

    return X, Y

def Label_Guessing(model, u1, u2):
    """ Pass k enhanced samples of unlabeled samples into the model, Then output average  probability distribution
    Args:
        model: output probability distribution
        u: k enhanced unlabeled samples (kxN)xCxHxW
    Return:
        q: Pseudo-label NxCxHxW
    """
    # storing the access of u 
    original_device = u1[0].device
    # without out gradient operation to speed up the processing
    with torch.no_grad():
        # the access of the model 
        u1 = u1.to(next(model.parameters()).device)
        u2 = u2.to(next(model.parameters()).device)
        # predict the u
        q1 = model(u1.float())['out']
        q2 = model(u2.float())['out']
        
        q = (torch.softmax(q1, 1) + torch.softmax(q2, 1)) /2

    return q


def Sharpening(p, T):
    """ make "pseudo" labels with lower entropy
    Args:
        p: Pseudo-label after Label_Guessing    NxCxHxW
        T: sharpening temperature
    Return:
        q: The label of the unlabeled sample    NxCxHxW
    """
    # every batch is taken the power of 1/T
    q = torch.pow(p,1/T)
    # take the summation along the N dimension.
    q = q/q.sum(dim = 1, keepdim =True)
    
    return q


def MixUp(X1, X2, Y1, Y2 , alpha):
    """ Mixup
    Args:
        X1: one Bacth Image, N
        X2: k Bactch images, K * N
        Y1: N
        Y2: K * N
        alpha : hyperparameter for beta distribution
    Return:
        X1_m: mixup images, labelled data
        X2_m: mixup images, unlabelled data
        Y1_m: mixup labels, label in labelled data
        Y2_m: mixup labels, predicted labelled in unlabelled data
    """
    # get weight from beta distribution
    alpha = torch.tensor(alpha, dtype=torch.float64)
    beta = torch.distributions.beta.Beta(alpha, alpha)
    weight = beta.sample().item()
    weight = max(weight, 1-weight)

    # compute W, combination labelled and unlabeld data
    indices = torch.randperm(len(X1) + len(X2))
    W_input = torch.cat((X1,X2),dim = 0)[indices]
    W_label = torch.cat((Y1,Y2),dim = 0)[indices]

    # mix up x and u
    # for labelled data pairs ：x1 and y1
    X = weight * X1 + (1 - weight) * W_input[:len(X1)]
    p = weight * Y1 + (1 - weight) * W_label[:len(X1)]

    # for unlabelled data pairs : x2 and y2
    U = weight * X2 + (1 - weight) * W_input[len(X1):]
    q = weight * Y2 + (1 - weight) * W_label[len(X1):]

    return X,U,p,q

def MixMatch(model, input_labelled, input_unlabelled, mask_labelled, T, K, alpha):
    """Refer to the pseudocode of mixmatch paper
    Args:
        model: deeplabv3 model
        input_labelled: one batch of images
        input_unlabelled: K batches of images
        mask_labelled: one batch of images' masks
        T: sharpening temperture
        K: number of augmentation
    Return:
        X: Mixmatch labelled images
        U: Mixmatch unlabelled images
        p: Mixmatch labelled masks
        q: Mixmatch predicted labelled masks
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # do Augmentation once for labelled images and masks
    X_Aug, mask_labelled_Aug = Data_Augmentation(input_labelled, mask_labelled)
    # do k Augmentation once for unlabelled images
    U_Aug1, _ = Data_Augmentation(input_unlabelled, mask_labelled)
    U_Aug2, _  = Data_Augmentation(input_unlabelled, mask_labelled)
    U_Aug = torch.cat([U_Aug1,U_Aug2], dim=0).to(torch.float32)

    # transform the mask from (N*256*256) to (N*3*256*256)
    labelled_mask = utils.transform_mask(mask_labelled_Aug)
    if torch.cuda.is_available():
      input_labelled = input_labelled.to(device)
      input_unlabelled = input_unlabelled.to(device)
      labelled_mask = labelled_mask.to(device)

    # Predict and sharpen the unlabel masks
    Guess_label = Label_Guessing(model, U_Aug1, U_Aug2)
    sharpen_label = Sharpening(Guess_label, T)
    mask_Unlabelled = torch.cat([sharpen_label for _ in range(K)], dim=0)
    

    X_Aug = X_Aug.to(device)
    U_Aug = U_Aug.to(device)
    mask_labelled = mask_labelled.to(device)
    mask_Unlabelled = mask_Unlabelled.to(device)
    mask_labelled_Aug = mask_labelled_Aug.to(device)

    # perform mixup
    X, U, p, q = MixUp(X_Aug, U_Aug, labelled_mask, mask_Unlabelled, alpha)

    return X,U,p,q