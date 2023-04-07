import numpy as np
from ncempy.io import dm
from glob import glob


def xray_correct_threshold(image,threshold=3000):
    if type(image) is not np.ndarray:
        raise TypeError('Input must be numpy ndarray.')
    image[image<0] = 0
    bad_loc = np.argwhere(image > (np.median(image)+threshold))
    bad_loc_vals = image[image > (np.median(image)+threshold)]
    print('Number of detected x-rays: '+ str(len(bad_loc)))
    print('Updated')
    
    #Organize pixels by intensity
    sorted_ind = np.argsort(bad_loc_vals) #indices of lowest -> highest bad pixels
    bad_loc_sorted = bad_loc[sorted_ind,:] 
    bad_loc_sorted = np.flip(bad_loc_sorted,axis=0) #Reverse the order so it goes highest -> lowest
    for loc in bad_loc_sorted:
        if loc[0]+1 > (image.shape[0]-1) or loc[0]-1 < 0 or loc[1]+1 > (image.shape[1]-1) or loc[1] - 1 < 0:
            #If x-ray is in the corners, let the pixel value be the mean
            new_pixel_int = image.mean()
        else:
            #Otherwise take average of all 8 pixels surrounding the pixel in question
            neighbor_sum = np.sum(image[loc[0]-1,(loc[1]-1):(loc[1]+2)])+ image[loc[0],loc[1]-1] + image[loc[0],loc[1]+1] + np.sum(image[loc[0]+1,(loc[1]-1):(loc[1]+2)])
            new_pixel_int = neighbor_sum /8
        image[loc[0],loc[1]] = new_pixel_int
    #Repeat once more if there are still outliers
    bad_loc = np.argwhere(image > (np.median(image)+threshold))
    bad_loc_vals = image[image > (np.median(image)+threshold)]
    print('Number of detected x-rays: '+ str(len(bad_loc)))
    if len(bad_loc) > 0:
        sorted_ind = np.argsort(bad_loc_vals) #indices of lowest -> highest bad pixels
        bad_loc_sorted = bad_loc[sorted_ind,:] 
        bad_loc_sorted = np.flip(bad_loc_sorted,axis=0) #Reverse the order so it goes highest -> lowest
        for loc in bad_loc_sorted:
            if loc[0]+1 > (image.shape[0]-1) or loc[0]-1 < 0 or loc[1]+1 > (image.shape[1]-1) or loc[1] - 1 < 0:
                #If x-ray is in the corners, let the pixel value be the mean
                new_pixel_int = image.mean()
            else:
                #Otherwise take average of all 8 pixels surrounding the pixel in question
                neighbor_sum = np.sum(image[loc[0]-1,(loc[1]-1):(loc[1]+2)])+ image[loc[0],loc[1]-1] + image[loc[0],loc[1]+1] + np.sum(image[loc[0]+1,(loc[1]-1):(loc[1]+2)])
                new_pixel_int = neighbor_sum /8
            image[loc[0],loc[1]] = new_pixel_int
    
    return image

def create_patches(images,labels,patch_size=512):
    #patch_size is the size of the patch (assume it will evenly go into the image)
    num_images = len(images)
    img_patches = []
    lbl_patches = []
    for i in range(num_images):
        img_to_split = images[i]
        lbl_to_split = labels[i]
        num_patch_y = img_to_split.shape[0]//patch_size
        num_patch_x = img_to_split.shape[1]//patch_size
        
        for j in range(num_patch_y):
            for k in range(num_patch_x):
                img_patches.append(img_to_split[j*patch_size:(j+1)*patch_size,k*patch_size:(k+1)*patch_size])
                lbl_patches.append(lbl_to_split[j*patch_size:(j+1)*patch_size,k*patch_size:(k+1)*patch_size])
    return img_patches, lbl_patches

    
def standardize(data_array):
    #Takes in a data array and sets its mean to 0 and std to 1
    mean = np.mean(data_array)
    std = np.std(data_array)
    return (data_array-mean)/std

    
def dihedral_augmentation(data_array,data_labels):
  #Takes in a (N,H,W,C) data array and its labels
  #Performs 90, 180, and 270 degree rotations, vertical flip, and horizontal flip. 
  #Returns data array and the expanded labels, though nothing is shuffled
  data_list = []
  label_list = []
  for i in range(data_array.shape[0]):
        image = data_array[i,:,:,:]
        label = data_labels[i,:,:,:]
        data_list.append(image)
        label_list.append(label)
        data_list.append(np.rot90(image,k=1))
        label_list.append(np.rot90(label,k=1))
        data_list.append(np.rot90(image,k=2))
        label_list.append(np.rot90(label,k=2))
        data_list.append(np.rot90(image,k=3))
        label_list.append(np.rot90(label,k=3))
        data_list.append(np.fliplr(image))
        label_list.append(np.fliplr(label))
        data_list.append(np.rot90(np.fliplr(image),k=1))
        label_list.append(np.rot90(np.fliplr(label),k=1))
        data_list.append(np.rot90(np.fliplr(image),k=2))
        label_list.append(np.rot90(np.fliplr(label),k=2))
        data_list.append(np.rot90(np.fliplr(image),k=3))
        label_list.append(np.rot90(np.fliplr(label),k=3))
  return np.asarray(data_list),np.asarray(label_list)

def shuffle_dataset(data_array, data_labels, seed):
    #Takes a (N,H,W,C) dataset and labels and the randomization seed
    #Shuffles accordingly
    #Returns shuffled arrays
    np.random.seed(seed)
    new_index = np.arange(0,data_array.shape[0],1)
    np.random.shuffle(new_index)
    return data_array[new_index,:,:,:], data_labels[new_index,:,:,:]

def pyTorch_format(data_array):
    #Takes in a (N,H,W,C) numpy data array in float64 data type
    #Converts to a (N,C,H,W) numpy data array in float32 data type for PyTorch processing
    data_array = data_array.transpose((0,3,1,2))
    return np.float32(data_array)
