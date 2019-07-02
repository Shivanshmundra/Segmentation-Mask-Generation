## Instructions for extraction of segmentation mask from pretrained Mask-RCNN model 

Inference model is based on Resnet101 backbone and is pre-trained on MS COCO dataset.

#### 				Prerequisites 

Code is intended to work with `Python 3.6.x` , it hasn't been tested with previous version. This code can run on CPU/GPU backend.

## Inference 

1. ```bash
   pip install -r requirements.txt
   ```

2. ```python
   python3 mask_generate.py --image_dir ./images --out_dir ./masks
   ```

   You can also specify which type of objects to segment out.

#### Acknowledgements 

Code is heavily borrowed from this repository : [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

I merely created a simple wrapper around it.