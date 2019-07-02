## Instructions for extraction of segmentation mask from pretrained Mask-RCNN model 

Inference model is based on Resnet101 backbone and is pre-trained on MS COCO dataset.

#### 				Prerequisites 

Code is intended to work with `Python 3.6.x` , it hasn't been tested with previous version. This code can run on CPU/GPU backend.

## Inference 

 ```bash
   pip install -r requirements.txt
   ```

```python
  python3 mask_generate.py --image_dir ./images --out_dir ./masks
   ```

   You can also specify which type of objects to segment out here [object_list = ['car', 'truck']](https://github.com/Shivanshmundra/Segmentation-Mask-Generation/blob/d21f76efdeeaf8a6f9ec384d7fc6aae21889304a/mask_generate.py#L25).

#### Acknowledgements 

Code is heavily borrowed from this repository : [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

I merely created a simple wrapper around it.

#### Input
![](https://github.com/Shivanshmundra/Segmentation-Mask-Generation/blob/master/images/cars.jpg)

#### Output 
![](https://github.com/Shivanshmundra/Segmentation-Mask-Generation/blob/master/masks/mask_cars.jpg)
