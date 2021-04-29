# Algonauts2021-devkit

This repository provides the code to predict human fMRI responses for videos from features of a baseline deep neural network (DNN) model (AlexNet). We further provide scripts to prepare the submission for the Algonauts 2021 challenge.  The terms features and activations are used interchangeably in the code.

**IMPORTANT: The Alexnet example is provided to explain how to prepare submissions using a baseline model. Please prepare your results in same format to participate in the challenge.!**

## Setup
* Install <a href="https://docs.anaconda.com/anaconda/install/">anaconda</a>
* Clone the repository ```git clone https://github.com/Neural-Dynamics-of-Visual-Cognition-FUB/Algonauts2021_devkit.git```
* Change working directory ```cd Algonauts2021_devkit```
* Run ```conda create -n Algonauts2021 python=3.7 anaconda``` to setup a new conda environment with required libraries
* Activate environment ```conda activate Algonauts2021```
* Install <a href="https://nilearn.github.io/introduction.html#installation">nilearn </a>, <a href="https://pytorch.org/">pytorch </a> and <a href="https://github.com/opencv/opencv-python">opencv </a>
* Download the data <a href="https://forms.gle/qq9uqqu6SwN8ytxQ9">here</a> and unzip in the working directory. Data is organized in two directories
   * AlgonautsVideos268_All_30fpsmax : contains 1102 videos: training (first 1000) and test (last 102) videos.
   * participants_data_v2021 : contains fMRI responses to training videos for both the challenge tracks.

## Step 1: Extract Alexnet features for the videos
* Run ```python feature_extraction/generate_features_alexnet.py``` to extract pretrained AlexNet features from the videos for all layers of AlexNet.
* With the default arguments, the script expects a directory ```./AlgonautsVideos268_All_30fpsmax/``` with the video sequences and saves the features in a directory called ````./alexnet````
* This code saves Alexnet features for every frame of every video, as well as a PCA transformation of these features to get top-100 components. These activations are split in train and test data
<details>
<summary>Arguments:</summary>

+ ```-vdir --video_data_dir```: Directory where the downloaded video sequences are stored (eg. ```./AlgonautsVideos268_All_30fpsmax/```)
+ ````-sdir --save_dir````: Directory where the exctracted features should be saved

</details>


## Step 2: Predict fMRI responses
* Run ```python perform_encoding.py``` to create predicted fMRI responses for test videos based on AlexNet features **or custom Neural Network Layers**
* With the default arguments, the script expects a directory ````./participants_data_v2021```` with real fMRI data and ````./alexnet/```` with extracted NN features. It will generate predicted features using Alexnet (````--model````) layer_5 (````--layer````) for EBA (````--roi````) of subject 4 (````--sub````) in validation mode (````--mode````). The results will be stored in a directory called ````./results````. Running the script in default mode should return a mean correlation of 0.23
* Example command for whole brain data (will take several minutes): ```python perform_encoding.py -rd ./results -ad ./alexnet/ -model alexnet_devkit -l layer_5 -sub sub01 -r WB -m val -fd ./participants_data_v2021 -v True -b 1000```

<details>
<summary>Arguments:</summary>

+ ````-rd --result_dir````: Result directory where the predicted fMRI activity will be saved
+ ````-ad --activation_dir````: Features directory, this directory should contain the DNN features for training the linear regression and predicting test fMRI data (eg. ```./alexnet``` after running Step 1)
+ ````-model --model````: Specify the model name, under which the results will be stored
+ ````-l --layer````: Specify the Neural Network layer to fit a linear mapping between activations and fMRI responses on training videos and predict test fMRI responses. For alexnet this should be ````layer_X```` with X between 1 and 8
+ ````-sub --sub````: Select the subject from which the fMRI data should be used to train (and validate) the linear Regression, for the fMRI data this should be ````subXX````with XX in (01, 02, 03, 04, 05, 06, 07, 08, 09, 10)
+ ````-r --roi````: Specify the region of interest (e.g. V1, LOC) from which fMRI data should be used; ```--roi WB``` uses the data from the Whole Brain
+ ````-m --mode````: Specify in which mode the program should run: "val": 10% of the original training data will be used as validation data. If in validation mode a mean correlation between the real fMRI response and the predicted fMRI response is also calculated; "test": All training data will be used for training
+ ````-fd --fmri_dir````: Directory which contains all recorded fMRI activity
+ ````-v --visualize````: Visualize correlations in the whole brain (True or False), only available if ````-roi WB````
+ ````-b --batch_size````: Set the number of voxels to fit in one iteration, default is 1000, reduce in case of memory constraints
</details>

* **Note: Predicted results should be generated for all combinations of ROIs and subjects.** We have given an example file to generate results for all ROIs and all subjects in a given track. Please run ```python generate_all_results.py``` to generate predicted results using default model and layer.
  * For ```mini_track``` please run ```python generate_all_results.py -t mini_track```
  * For ```full_track``` please run ```python generate_all_results.py -t full_track```
  
## Step 3: Prepare Submission
* After generating predicted fMRI activity for **all** the ROIs and **all** the subjects in a given track (```mini_track``` and/or ```full_track```) using generate_all_results.py, run ```python prepare_submission.py``` in order to prepare the results for submission. All results will be combined into a single file.
  * For ```mini_track``` please run ```python prepare_submission.py -t mini_track```
  * For ```full_track``` please run ```python prepare_submission.py -t full_track```
* With the default arguments, the script expects the results from step 2 in a directory ```./results/alexnet_devkit/layer_5```. It prepares the submission for all 9 ROIs (```mini_track```) . To generate results for ```full_track``` change the arguments as mentioned above.
* The script creates a Pickle and a zip file (containing the Pickle file) for the corresponding track that can then be submitted for participation in the challenge.
* Submit the ```mini_track``` results <a href="https://competitions.codalab.org/competitions/30930?secret_key=0d92787c-69d7-4e38-9780-94dd3a301f6b#participate-submit_results">here</a> and ```full_track``` results <a href="https://competitions.codalab.org/competitions/30937?secret_key=f3d0f352-c582-49cb-ad7c-8e6ec9702054#participate-submit_results">here</a> 

<details>
<summary>Arguments:</summary>

+ ````-rd --result_dir````: Directory containing the predicted fMRI activity from step 2, should be identical to result_dir there.
+ ````-t --track````: ```mini_track``` for the specific ROIs, ```full_track``` for whole brain (WB) data. Submission can be done for either one of them separately, for submitting both the submission script should be run twice, once with ````mini_track```` and once with ````full_track````

</details>

## Cite

If you use our code, partly or as is,  please cite the paper below

```
@misc{cichy2021algonauts,
      title={The Algonauts Project 2021 Challenge: How the Human Brain Makes Sense of a World in Motion}, 
      author={R. M. Cichy and K. Dwivedi and B. Lahner and A. Lascelles and P. Iamshchinina and M. Graumann and A. Andonian and N. A. R. Murty and K. Kay and G. Roig and A. Oliva},
      year={2021},
      eprint={2104.13714},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
