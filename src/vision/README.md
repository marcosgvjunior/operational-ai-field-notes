# Computer Vision (`src/vision`)

This sub-directory contains all computer vision related modules.

## Modules

*   `boxes.py`: Defines the primary `Box` data structure and the core "operational contract" functions, including Intersection over Union (`iou`) and Non-Maximum Suppression (`nms`).
*   `contracts.py`: Defines the data contracts (e.g. `DetectionResult`) for consistent data structures across different models.
*   `segmentation.py`: An adapter module for `torchvision` semantic segmentation models.
*   `tfhub_det.py`: An adapter module for TensorFlow Hub object detection models.
*   `tfhub_det_openimages.py`: An adapter module containing a wrapper for a specific TensorFlow Hub object detection model (SSD w/ MobileNetV2) trained on the Open Images V4 dataset.
*   `torchvision_det.py`: An adapter module for PyTorch/Torchvision object detection models.
*   `viz.py`: Contains utility functions for drawing bounding boxes, labels, and scores on images to visualize model outputs.
*   `yolo_ultralytics_det.py`: An adapter module for Ultralytics YOLO models.
