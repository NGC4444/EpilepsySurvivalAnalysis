import os
import numpy as np
import nibabel as nib
import pickle as pkl
import torch

cachefile = np.load('/media/user/hd1/shaobo/cache/data1_wholeslide_mri_part1.npy',allow_pickle=True).item()
cachefile2 = np.load('/media/user/hd1/shaobo/cache/data1_wholeslide_mri_part2.npy',allow_pickle=True).item()
# cachefile3 = np.load('/home/user/disk/shaobo/drwho/data/cache/data3_wholeslide_twoside_part3.npy',allow_pickle=True).item()
# cachefile4 = np.load('/home/user/disk/shaobo/drwho/data/cache/data3_wholeslide_twoside_part4.npy',allow_pickle=True).item()
# cachefile5 = np.load('/home/user/disk/shaobo/drwho/data/cache/data3_wholeslide_twoside_part5.npy',allow_pickle=True).item()
def load_PET_(id, phase='train'):
    root_train = '/home/user/disk/yang/TLE_nor/' # PET/T1/MASK5
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
        # elif f in cachefile3 and phase == 'train':
        #     used_roi.append(cachefile3[f])
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
def load_PET(id, phase='train'):
    root_train = '/media/user/hd1/yang/dataset/TLE_nor/'# PET/T1/MASK5
    root = root_train
    
    
    file_list = os.listdir(root)
    # print(len(file_list))
    used_roi = []
    unused_roi = []

    for f in id:
        if f in cachefile :
            used_roi.append(cachefile[f])
        elif f in cachefile2 :
            used_roi.append(cachefile2[f])
        # elif f in cachefile3 :
        #     used_roi.append(cachefile3[f])
        # elif f in cachefile4 :
        #     used_roi.append(cachefile4[f])
        # elif f in cachefile5 :
        #     used_roi.append(cachefile5[f])
        # elif f in file_list:
            
        #     pet_path = os.path.join(root,f,'PET')
        #     print("{} NOT IN CACHE!".format(id))
        #     pet_path = os.path.join(pet_path,os.listdir(pet_path)[0])
        #     mask_path = os.path.join(root,f,'MASK')
        #     mask_path = os.path.join(mask_path,'newmask.nii')
            
        #     pet_img = nib.load(pet_path).get_fdata()
        #     mask_img = nib.load(mask_path).get_fdata()
        #     used = pet_img * mask_img
        #     # notused = pet_img * (1 - mask_img)
        #     used = np.resize(used,(used.shape[0]//2,used.shape[1]//2,used.shape[2]))
        #     used_roi.append(used[:,:,:])
        #     # images.append(pet_img)
        #     # return used, notused
        else:
            print("{} NOT FOUND!".format(id))
            assert f in id
    return torch.tensor(used_roi), None
if __name__=='__main__':
    root = '/media/user/hd1/yang/dataset/TLE_nor' # PET/T1/MASK
    tublar_path = '2014-2019TLE220-CSB.csv'
    file_list = os.listdir(root)
    print(len(file_list))
    dic = {}
    i = 0
    used_roi = []
    for f in file_list:
        i += 1
        if(  i>100 ):
            pet_path = os.path.join(root,f,'PET')
            mri_path = os.path.join(root,f,'T1')
            # print(pet_path)
            pet_path = os.path.join(pet_path,os.listdir(pet_path)[0])
            mri_path = os.path.join(mri_path,os.listdir(mri_path)[0])

            mask_path = os.path.join(root,f,'MASK')
            mask_path = os.path.join(mask_path,'newmask.nii')
            
            used = nib.load(pet_path).get_fdata()
            mri = nib.load(mri_path).get_fdata()
            # mask_img = nib.load(mask_path).get_fdata()
            # used = pet_img #* mask_img
            # notused = pet_img * (1 - mask_img)
            used = mri
            used = np.resize(used,(used.shape[0]//2, used.shape[1]//2, used.shape[2]))
            # used_roi.append(used[:,:,8:74])
            # aff =nib.load(pet_path).affine
            # nib.Nifti1Image(pet_img[256:,:,:],aff).to_filename('exp1left.nii.gz')#左半边
            print("{} : {}".format(f, used.shape))
            dic[f] = used

    np.save("/media/user/hd1/shaobo/cache/data1_wholeslide_mri_part2.npy",dic)
    print('ok')



        

    