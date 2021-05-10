import os
import numpy as np
from utils.helper import make_path, load_dict
import cv2

def load_fmri(fmri_dir, subject, ROI):
    """This function loads fMRI data into a numpy array for to a given ROI.

    Parameters
    ----------
    fmri_dir : str
        path to fMRI data.
    subject : int or str
        subject number if int, 'sub#num' if str
    ROI : str
        name of ROI.

    Returns
    -------
    np.array
        matrix of dimensions #train_vids x #repetitions x #voxels
        containing fMRI responses to train videos of a given ROI
    """
    ROI_data = load_dict(make_path(fmri_dir, subject, ROI))
    return ROI_data["train"]

def load_voxel_mask_wb(fmri_dir, subject):
    """ Load voxel mask for whole brain fMRI data.

    Parameters
    ----------
    fmri_dir : str
        path to fMRI data.
    subject : int or str
        subject number if int, 'sub#num' if str

    Returns
    -------
    np.array
        matrix of dimensions #train_vids x #repetitions x #voxels
        containing fMRI responses to train videos of a given ROI
    """
    ROI_data = load_dict(make_path(fmri_dir, subject, ROI))
    return ROI_data['voxel_mask']

def load_video(file, frame_skip_step=1):
    """This function takes a video file as input and returns
    an array of frames in numpy format.

    Parameters
    ----------
    file : str
        path to video file
    frame_skip_step : int
        take every `frame_skip_step`th frame

    Returns
    -------
    video: np.array
        shape: (1, #num_frames, height, width, 3)
    """
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((int(frameCount / frame_skip_step), frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while fc < frameCount and ret:
        fc += 1
        if fc % frame_skip_step == 0:
            (ret, buf[int((fc - 1) / frame_skip_step)]) = cap.read()

    cap.release()
    return np.expand_dims(buf, axis=0)

def load_activations(activations_dir, layer_name):
    """This function loads neural network features/activations (preprocessed using PCA) into a
    numpy array according to a given layer.

    Parameters
    ----------
    activations_dir : str
        Path to PCA processed Neural Network features
    layer_name : str
        which layer of the neural network to load,

    Returns
    -------
    train_activations : np.array
        matrix of dimensions #train_vids x #pca_components
        containing activations of train videos
    test_activations : np.array
        matrix of dimensions #test_vids x #pca_components
        containing activations of test videos

    """

    train_file = os.path.join(activations_dir,"train_" + layer_name + ".npy")
    test_file = os.path.join(activations_dir,"test_" + layer_name + ".npy")
    train_activations = np.load(train_file)
    test_activations = np.load(test_file)
    scaler = StandardScaler()
    train_activations = scaler.fit_transform(train_activations)
    test_activations = scaler.fit_transform(test_activations)

    return train_activations, test_activations

vp = "data/AlgonautsVideos268_All_30fpsmax/0001_0-0-1-6-7-2-8-0-17500167280.mp4"

v = load_video(vp)
