from subprocess import Popen
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
from utils.helper import save_dict,load_dict



def main():

    parser = argparse.ArgumentParser(description='Generates predictions for all subs all ROIs for a given track')
    parser.add_argument('-t','--track', help='mini_track for all ROIs, full_track for whole brain (WB)', default = 'mini_track', type=str)
    parser.add_argument('-fd','--fmri_dir',help='directory containing fMRI activity', default = './data/participants_data_v2021', type=str)

    args = vars(parser.parse_args())
    track = args['track']
    fmri_dir = args['fmri_dir']

    if track == 'full_track':
        ROIs = ['WB']
    else:
        ROIs = ['LOC','FFA','STS','EBA','PPA','V1','V2','V3','V4']

    num_subs = 10
    subs=[]
    for s in range(num_subs):
      subs.append('sub'+str(s+1).zfill(2))


    for roi in ROIs:
      for sub in subs:
          cmd_string = 'python perform_encoding.py' + ' --roi ' + roi +  ' --sub ' + sub + ' -fd ' + fmri_dir + ' --mode  test'
          print("----------------------------------------------------------------------------")
          print("----------------------------------------------------------------------------")
          print ("Starting ROI: ", roi, "sub: ",sub)
          os.system(cmd_string)
          print ("Completed ROI: ", roi, "sub: ",sub)
          print("----------------------------------------------------------------------------")
          print("----------------------------------------------------------------------------")



if __name__ == "__main__":
    main()
