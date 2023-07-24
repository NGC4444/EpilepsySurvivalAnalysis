import os
import numpy as np
import nibabel as nib
import pickle as pkl
import torch

cachefile = np.load('data/cache/used_all_wholeslide.npy',allow_pickle=True).item()
cachefile2 = np.load('data/cache/used_all2_wholeslide.npy',allow_pickle=True).item()
def load_PET(id, phase='train'):
    root_train = '/home/user/disk/yang/TLE_nor/' # PET/T1/MASK
    root_test = "/home/user/disk/yang/MRINeg98_nor/"

    tublar_path = '2014-2019TLE220-CSB.csv'
    root = root_train
    phase = 'train'
    if phase == 'train':
        root = root_train
    else:
        root = root_test
    
    file_list = os.listdir(root)
    # print(len(file_list))
    used_roi = []
    unused_roi = []

    for f in id:
        if f in cachefile and phase == 'train':
            used_roi.append(cachefile[f])
        elif f in cachefile2 and phase == 'train':
            used_roi.append(cachefile2[f])
        elif f in file_list:
            
            pet_path = os.path.join(root,f,'PET')
            # print(pet_path)
            pet_path = os.path.join(pet_path,os.listdir(pet_path)[0])
            mask_path = os.path.join(root,f,'MASK')
            mask_path = os.path.join(mask_path,os.listdir(mask_path)[0])
            
            pet_img = nib.load(pet_path).get_fdata()
            mask_img = nib.load(mask_path).get_fdata()
            used = pet_img #* mask_img
            # notused = pet_img * (1 - mask_img)
            used = np.resize(used,(used.shape[0]//2,used.shape[1]//2,used.shape[2]))
            used_roi.append(used[:,:,:])
            # images.append(pet_img)
            # return used, notused
        else:
            print("{} NOT FOUND!".format(id))
    return torch.tensor(used_roi), None

if __name__=='__main__':
    root = '/home/user/disk/yang/TLE_nor/' # PET/T1/MASK
    tublar_path = '2014-2019TLE220-CSB.csv'
    file_list = os.listdir(root)
    print(len(file_list))
    dic = {}
    i = 0
    used_roi = []
    for f in file_list:
        i += 1
        if i >75:
            #<= 150----------------------------->   >150
            # i <= 75 and i>150 ----------------->  75-150
            # i > 75 --------------------------->  0-75
            continue
        pet_path = os.path.join(root,f,'PET')
        # print(pet_path)
        pet_path = os.path.join(pet_path,os.listdir(pet_path)[0])
        mask_path = os.path.join(root,f,'MASK')
        mask_path = os.path.join(mask_path,os.listdir(mask_path)[0])
        
        pet_img = nib.load(pet_path).get_fdata()
        mask_img = nib.load(mask_path).get_fdata()
        used = pet_img #* mask_img
        # notused = pet_img * (1 - mask_img)
        used = np.resize(used,(used.shape[0]//2,used.shape[1]//2,used.shape[2]))
        # used_roi.append(used[:,:,8:74])
        # aff =nib.load(pet_path).affine
        # nib.Nifti1Image(pet_img[256:,:,:],aff).to_filename('exp1left.nii.gz')#左半边
        dic[f] = used

    # with open('PET_220_used.pkl','wb') as fo:
    #     pkl.dump(dic,fo)
    #     # fo.close()
    np.save("used_all_wholeslide_allslide3.npy",dic)



        

    