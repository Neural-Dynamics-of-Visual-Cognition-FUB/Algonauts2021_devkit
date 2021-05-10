import glob
from feature_extraction.alexnet import load_alexnet
import numpy as np
import urllib
import torch
import argparse
import random
from tqdm import tqdm
from torchvision import transforms as trn
import os
from torch.autograd import Variable as V
from load import load_video

seed = 42
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)


def get_activations_and_save(model, video_list, activations_dir, frame_skip_step = 4):
    """This function generates Alexnet features for every frame in a video and save them in a specified directory.

    Parameters
    ----------
    model :
        pytorch model : alexnet.
    video_list : list
        the list contains path to all videos.
    activations_dir : str
        save path for extracted features.
    frame_skip_step : int
        take every `frame_skip_step`th frame
    """

    centre_crop = trn.Compose([
            trn.ToPILImage(),
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for video_file in tqdm(video_list):
        vid = load_video(video_file, frame_skip_step)
        activations = []
        for frame in range(vid.shape[1]):
            img =  vid[0,frame,:,:,:]
            input_img = V(centre_crop(img).unsqueeze(0))
            if torch.cuda.is_available():
                input_img = input_img.cuda()
            all_features = model.forward(input_img)
            if len(activations) == 0:
                activations.extend([[] for x in all_features])
            for i,x in enumerate(all_features):
                activations[i].append(x.detach().numpy())
            #activations.append([l.detach().numpy() for l in x])
            """
            for i,feat in enumerate(x):
                if frame==0:
                    activations.append(feat.data.cpu().numpy().ravel())
                else:
                    activations[i] =  activations[i] + feat.data.cpu().numpy().ravel()
            """
        #activations = np.array(activations)
        video_file_name = os.path.split(video_file)[-1].split(".")[0]
        for layer in range(len(activations)):
            save_path = os.path.join(activations_dir, video_file_name+"_"+"layer" + "_" + str(layer+1) + ".npy")
            #np.save(save_path,activations[layer])

def main():
    parser = argparse.ArgumentParser(description='Feature Extraction from Alexnet.')
    parser.add_argument('-vdir','--video_data_dir', help='video data directory',default = './data/AlgonautsVideos268_All_30fpsmax/', type=str)
    parser.add_argument('-sdir','--save_dir', help='saves processed features',default = './data/alexnet_frames', type=str)
    args = vars(parser.parse_args())

    save_dir=args['save_dir']
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    video_dir = args['video_data_dir']
    video_list = glob.glob(video_dir + '/*.mp4')
    video_list.sort()
    print('Total Number of Videos: ', len(video_list))

    # load Alexnet
    # Download pretrained Alexnet from:
    # https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
    # and save in the current directory
    checkpoint_path = "./alexnet.pth"
    if not os.path.exists(checkpoint_path):
        url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
        urllib.request.urlretrieve(url, "./alexnet.pth")
    model = load_alexnet(checkpoint_path)

    # get and save activations
    activations_dir = os.path.join(save_dir)
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)
    print("-------------Saving activations ----------------------------")
    get_activations_and_save(model, video_list, activations_dir)


if __name__ == "__main__":
    main()
