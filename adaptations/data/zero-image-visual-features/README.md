# Extract image features from a black images

## Extract CLIP features from a black image

The CLIP features are simply extracted in the [adaptations/src/generate_visual_features.ipynb](adaptations/src/generate_visual_features.ipynb) notebook. You can also use the already generated features in [adaptations/data/zero-image-visual-features/clip_features.pt](adaptations/data/zero-image-visual-features/clip_features.pt). 

## Extract Faster-R CNN detections from a black image (zero image)

We use the pre-trained Faster R-CNN model and Dockerfile provided in the LMXERT [repo](https://github.com/airsplay/lxmert) to get the Faster-R CNN features for a black image. Please follow their instructions on how to setup the image feature extractor. Alternatively, you can use the already generated [image features](output.csv).

The Faster-R CNN output is then formatted some more in the [adaptations/src/generate_visual_features.ipynb](adaptations/src/generate_visual_features.ipynb) notebook to get the corresponding frcnn detection features and boxes. These are also already generated at [adaptations/data/zero-image-visual-features/frcnn_boxes.pt](adaptations/data/zero-image-visual-features/frcnn_boxes.pt) and [adaptations/data/zero-image-visual-features/frcnn_features.pt](adaptations/data/zero-image-visual-features/frcnn_features.pt).

### Generate Faster-R CNN features for a black image

The steps below assume that you have:
    * Downloaded the docker file `airsplay/bottom-up-attention`
    * Downloaded the necessary model weights `resnet101_faster_rcnn_final_iter_320000.caffemodel` 

Make sure to stand in this directory when running the following commands.

```bash
docker run --gpus all -v zero-image-visual-features:/workspace --rm -it airsplay/bottom-up-attention bash
python extract_image_detections.py --im_file filled_with_0.png
```

The `output.csv` file will then be the Faster-R CNN features for the black image. 