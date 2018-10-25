#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

import numpy as np
import sys
import os, os.path
# %matplotlib inline
import matplotlib.image as mpimg

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
    # Brightness 
    

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    img = augment_brightness_camera_images(img)
    
    return img

_, dirs, _ = next(os.walk("data/train_images"))
max = -5
min = 10000
sum =0 
for i in range(len(dirs)):
    _, _, images = next(os.walk("data/train_images/" + dirs[i]))
    print(len(images))
    sum = sum + len(images)
    if len(images) > max:
        max =len(images)
    if len(images) < min:
        min =len(images)
print(sum)
print(max)
print(min)

print("lets see")

# minimum_required = int(0.8*max)
# for direct in range(len(dirs)):
#     _, _, images = next(os.walk("data/train_images/" + dirs[direct]))
#     if len(images) < minimum_required:
#         imagesCount = len(images)
#         # shouldBreak = False 
#         # newTotalCount = imagesCount
#         per_image_augment  = int(minimum_required/imagesCount)
#         print(per_image_augment)
#         for img in range(imagesCount):
#             image = mpimg.imread('data/train_images/' + dirs[direct] +'/' +images[img])
#             name  = images[img].split('.')[0]
#             for i in range(per_image_augment):
#                 # if newTotalCount > minimum_required:
#                     # shouldBreak = True
#                     # break
#                 # newTotalCount = newTotalCount + 1;
#                 img = transform_image(image,20,10,5)
#                 path = 'data/train_images/' + dirs[direct]
#                 cv2.imwrite(os.path.join(path , name + '_augment_'+str(i)+'.ppm') , cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#             # if shouldBreak:
#                 # break
        # print(newTotalCount)  
