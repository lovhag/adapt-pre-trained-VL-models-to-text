# avg-visual-features

This folder should contain the visual features that are the average of the train and validation data of each of the vision-and-language models evaluated (CLIP-BERT, LXMERT and VisualBERT):
* `frcnn_features_per_detection.pt` and `frcnn_boxes_per_detection.pt`
* `clip_features.pt`

These features are generated using the [adaptations/src/generate_visual_features.ipynb](adaptations/src/generate_visual_features.ipynb) notebook.