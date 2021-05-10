from os.path import join
import nibabel as nib
import pickle

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
        #print(p)
        #ret_di = pickle.load(f)
    return ret_di

def saveasnii(brain_mask,nii_save_path,nii_data):
    img = nib.load(brain_mask)
    print(img.shape)
    nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
    nib.save(nii_img, nii_save_path)

def make_path(fmri_dir, subject, ROI):
    """ Create path to data file for given base dir, subject and ROI.

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
    str
        path to a data file
    """
    track = "full_track" if ROI == "WB" else "mini_track"
    if type(subject) is int:
        subject = f"sub{subject:02}"
    return join(fmri_dir, track, subject, ROI + ".pkl")

