import numpy as np
import matplotlib.pyplot as plt
from load import load_fmri
from sklearn.decomposition import PCA

data_dir = "./data/participants_data_v2021/"
subjects = [f"sub{i:02}" for i in range(1,11)]
rois = ["V1", "V2", "V3", "V4", "EBA", "FFA", "LOC", "PPA", "STS"]

subjects = [subjects[3]]
rois = [rois[0]]

for s in subjects:
    for r in rois:
        fmri = load_fmri(data_dir, s, r)
        fmri_flat = fmri.reshape(-1, fmri.shape[2])
        pca = PCA(n_components=2)
        pca.fit(fmri_flat)
        trans = pca.transform(fmri_flat)
        print(trans.shape)

        for i in range(10):
            plt.scatter(trans[i*3:i*3+3,0], trans[i*3:i*3+3,1])
            #plt.scatter(trans[3:6,0], trans[3:6,1])
        plt.title(f"{s}  {r}")
        plt.show()

