import os
import sys
import random
import math
import numpy as np
import skimage.io
from PIL import Image
from tqdm import tqdm
import argparse

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from coco_config import *

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='/silocloud/buckets/silo_data/train/A', help='Image dir for mask generation')
parser.add_argument('--root_dir', type=str, default='./', help='root directory of project')
parser.add_argument('--out_dir', type=str, default='/silocloud/buckets/silo_data/mask/A', help='directory of mask to save')
parser.add_argument('--model_dir', type=str, default="./mask_rcnn_coco.h5", help='Local path to trained weights file')
parser.add_argument('--object_list', type=list, default=['car', 'truck'], help='objects to segment, to be provided as list of str, should be from class_list')
parser.add_argument('--is_resize', type=bool, default=False, help='to resize segmented mask')
parser.add_argument('--resize_dim', type=int, default=256, help='size of the data resize (squared assumed)')

args = parser.parse_args()
print(args)

# Download COCO trained weights from Releases if needed
if not os.path.exists(args.model_dir):
    utils.download_trained_weights(args.model_dir)

# Directory of images to run detection on
# IMAGE_DIR = '/silocloud/buckets/silo_data/train/A'


# MASK_DIR = '/silocloud/buckets/silo_data/mask/A'
if not os.path.exists(args.out_dir):
	os.makedirs(args.out_dir)


config = InferenceConfig()
print('Configuration Input is')
print(config)

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=args.root_dir, config=config)

# Load weights trained on MS-COCO
model.load_weights(args.model_dir, by_name=True)


# get file names from image dir to load
file_names= [file for file in os.listdir(args.image_dir) if file.endswith('.png')]

# get id list from class of objects
id_list = [class_names.index(name) for name in args.object_list]

# Run detection and save masks
for file_name in tqdm(file_names):
    image = skimage.io.imread(os.path.join(args.image_dir, file_name))
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    results = model.detect([image])
    class_ids = results[0]['class_ids']
    for id_ in range(len(class_ids)):
        if class_ids[id_] in id_list:
            mask += results[0]['masks'][:,:,id_]
    mask[mask > 0] = 255 
    img = Image.fromarray(mask)
    if args.is_resize:
    	img = img.resize((args.resize_dim,args.resize_dim), Image.ANTIALIAS)
    img.save(os.path.join(args.out_dir, 'mask_' + file_name))
    
