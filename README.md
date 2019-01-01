# car-dl
 There is two different models in this repo and sample outputs of both in the samples folder.

# Bounding Box Model
 The first model is found in the root of the repo. It uses VGG16 transfer learning and L1 loss to compute the bounding box for the input image.

![Image 1](https://raw.githubusercontent.com/Troy-M/car-dl/master/samples/bounding.png)

 # Mask Model
 The second model is found in the mask folder. It uses a U-Net model evaluated with dice coefficent. This model works extremely well on this dataset and is in the high 90%s.

![Image 2](https://raw.githubusercontent.com/Troy-M/car-dl/master/samples/276.png)
