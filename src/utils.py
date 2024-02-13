import os
import sys
import torch
from PIL import Image
from exceptions import CustomException
from logger import logging

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not(os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except Exception as e:
        logging.exception(e)
        raise CustomException(e,sys)

def check_model_exists(model_path: str)-> bool:
    try:
        if os.path.isfile(model_path):
            return True
        else:
            return False
    except Exception as e:
        logging.exception(e)
        raise CustomException(e,sys)

def load_image(filename, size = None, scale = None):
    '''
    Load an image from the given filename and perform required scaling and resizing
    '''
    try:
        img = Image.open(filename)
        if size is not None:
            img = img.resize((size, size), Image.ANTIAlIAS)
        elif scale is not None:
            img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
        return img
    except Exception as e:
        logging.exception(e)
        raise CustomException(e,sys)


def save_image(filename, data):
    '''
    save an image from the given data(tensor) to the specified filename
    Params:
        filename: str -> path where to save the image
        data: tensor -> input tensor which contains the image information
    Returns:
        None
    '''
    try:
        img = data.clone().clamp(0,255).numpy()
        img = img.transpose(1,2,0).astype("uint8")
        img = Image.fromarray(img)
        img.save(filename)
    except Exception as e:
        logging.exception(e)
        raise CustomException(e,sys)


def gram_matrix(y):
    '''
    Computes gram matrix of features in y.
    Params:
        y: 4D tensor
    Returns:
        Gram matrix of features in y
    '''
    try:
        (b,ch,h,w) = y.size()
        features = y.view(b, ch, w*h)
        features_T = features.transpose(1, 2)
        G = features.bmm(features_T) / (ch * h * w)
        return G
    except Exception as e:
        logging.exception(e)
        raise CustomException(e,sys)


def normalize_batch(batch):
    #normalize using imagenet mean and std
    try:
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        batch = batch.div_(255.0)
        batch -= mean
        batch /= std
        return batch
    except Exception as e:
        logging.exception(e)
        raise CustomException(e,sys)

