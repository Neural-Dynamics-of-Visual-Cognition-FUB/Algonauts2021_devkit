# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import random
import argparse
import itertools
import nibabel as nib
from nilearn import plotting
from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import time
import pickle
from tqdm import tqdm
from utils.ols import vectorized_correlation,OLS_pytorch
from utils.helper import save_dict,load_dict, saveasnii

def get_activations(activations_dir, layer_name):
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

def get_fmri(fmri_dir, ROI):
    """This function loads fMRI data into a numpy array for to a given ROI.

    Parameters
    ----------
    fmri_dir : str
        path to fMRI data.
    ROI : str
        name of ROI.

    Returns
    -------
    np.array
        matrix of dimensions #train_vids x #repetitions x #voxels
        containing fMRI responses to train videos of a given ROI

    """


    # Loading ROI data
    ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    # averaging ROI data across repetitions
    ROI_data_train = np.mean(ROI_data["train"], axis = 1)
    if ROI == "WB":
        voxel_mask = ROI_data['voxel_mask']
        return ROI_data_train, voxel_mask

    return ROI_data_train

def predict_fmri_fast(train_activations, test_activations, train_fmri,use_gpu=False):
    """This function fits a linear regressor using train_activations and train_fmri,
    then returns the predicted fmri_pred_test using the fitted weights and
    test_activations.

    Parameters
    ----------
    train_activations : np.array
        matrix of dimensions #train_vids x #pca_components
        containing activations of train videos.
    test_activations : np.array
        matrix of dimensions #test_vids x #pca_components
        containing activations of test videos
    train_fmri : np.array
        matrix of dimensions #train_vids x  #voxels
        containing fMRI responses to train videos
    use_gpu : bool
        Description of parameter `use_gpu`.

    Returns
    -------
    fmri_pred_test: np.array
        matrix of dimensions #test_vids x  #voxels
        containing predicted fMRI responses to test videos .

    """

    reg = OLS_pytorch(use_gpu)
    reg.fit(train_activations,train_fmri.T)
    fmri_pred_test = reg.predict(test_activations)
    return fmri_pred_test

def main():

    parser = argparse.ArgumentParser(description='Encoding model analysis for Algonauts 2021')
    parser.add_argument('-rd','--result_dir', help='saves predicted fMRI activity',default = './results', type=str)
    parser.add_argument('-ad','--activation_dir',help='directory containing DNN activations',default = './alexnet/', type=str)
    parser.add_argument('-model','--model',help='model name under which predicted fMRI activity will be saved', default = 'alexnet_devkit', type=str)
    parser.add_argument('-l','--layer',help='layer from which activations will be used to train and predict fMRI activity', default = 'layer_5', type=str)
    parser.add_argument('-sub','--sub',help='subject number from which real fMRI data will be used', default = 'sub04', type=str)
    parser.add_argument('-r','--roi',help='brain region, from which real fMRI data will be used', default = 'EBA', type=str)
    parser.add_argument('-m','--mode',help='test or val, val returns mean correlation by using 10% of training data for validation', default = 'val', type=str)
    parser.add_argument('-fd','--fmri_dir',help='directory containing fMRI activity', default = './participants_data_v2021', type=str)
    parser.add_argument('-v','--visualize',help='visualize whole brain results in MNI space or not', default = True, type=bool)
    parser.add_argument('-b', '--batch_size',help=' number of voxel to fit at one time in case of memory constraints', default = 1000, type=int)
    args = vars(parser.parse_args())


    mode = args['mode'] # test or val
    sub = args['sub']
    ROI = args['roi']
    model = args['model']
    layer = args['layer']
    visualize_results = args['visualize']
    batch_size = args['batch_size'] # number of voxel to fit at one time in case of memory constraints

    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    if ROI == "WB":
        track = "full_track"
    else:
        track = "mini_track"

    activation_dir = os.path.join(args['activation_dir'],'pca_100')
    fmri_dir = os.path.join(args['fmri_dir'], track)

    sub_fmri_dir = os.path.join(fmri_dir, sub)
    results_dir = os.path.join(args['result_dir'],args['model'], args['layer'], track, sub)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("ROi is : ", ROI)

    train_activations,test_activations = get_activations(activation_dir, layer)
    if track == "full_track":
        fmri_train_all,voxel_mask = get_fmri(sub_fmri_dir,ROI)
    else:
        fmri_train_all = get_fmri(sub_fmri_dir,ROI)
    num_voxels = fmri_train_all.shape[1]
    if mode == 'val':
        # Here as an example we use first 900 videos as training and rest of the videos as validation
        test_activations = train_activations[900:,:]
        train_activations = train_activations[:900,:]
        fmri_train = fmri_train_all[:900,:]
        fmri_test = fmri_train_all[900:,:]
        pred_fmri = np.zeros_like(fmri_test)
        pred_fmri_save_path = os.path.join(results_dir, ROI + '_val.npy')
    else:
        fmri_train = fmri_train_all
        num_test_videos = 102
        pred_fmri = np.zeros((num_test_videos,num_voxels))
        pred_fmri_save_path = os.path.join(results_dir, ROI + '_test.npy')


    print("number of voxels is ", num_voxels)
    iter = 0
    while iter < num_voxels-batch_size:
        pred_fmri[:,iter:iter+batch_size] = predict_fmri_fast(train_activations,test_activations,fmri_train[:,iter:iter+batch_size], use_gpu = use_gpu)
        iter = iter+batch_size
        print((100*iter)//num_voxels," percent complete")
    pred_fmri[:,iter:] = predict_fmri_fast(train_activations,test_activations,fmri_train[:,iter:iter+batch_size], use_gpu = use_gpu)

    if mode == 'val':
        score = vectorized_correlation(fmri_test,pred_fmri)
        print("----------------------------------------------------------------------------")
        print("Mean correlation for ROI : ",ROI, "in ",sub, " is :", round(score.mean(), 3))

        # result visualization for whole brain (full_track)
        if track == "full_track" and visualize_results:
            visual_mask_3D = np.zeros((78,93,71))
            visual_mask_3D[voxel_mask==1]= score
            brain_mask = './example.nii'
            nii_save_path =  os.path.join(results_dir, ROI + '_val.nii')
            saveasnii(brain_mask,nii_save_path,visual_mask_3D)
            view = plotting.view_img_on_surf(nii_save_path, threshold=None, surf_mesh='fsaverage',\
                                            title = 'Correlation for sub' + sub, colorbar=False)
            view_save_path = os.path.join(results_dir,ROI + '_val.html')
            view.save_as_html(view_save_path)
            print("Results saved in this directory: ", results_dir)
            view.open_in_browser()


    np.save(pred_fmri_save_path, pred_fmri)


    print("----------------------------------------------------------------------------")
    print("ROI done : ", ROI)

if __name__ == "__main__":
    main()
