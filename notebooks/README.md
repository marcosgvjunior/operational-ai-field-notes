# Notebooks

This directory contains the primary deliverable of the project: a series of Jupyter Notebooks that serve as practical, hands-on demonstrations.

## Purpose & Philosophy

The notebooks are designed to be educational "field notes" that show how the abstract, reusable logic from the `../src/` directory is applied to real-world models from popular machine learning frameworks.

They are not just code dumps; they follow a narrative structure, explaining each step of the process:
1.  Loading a pre-trained model.
2.  Preparing an input image.
3.  Running inference to get raw model outputs.
4.  **Applying the "operational contract"** from `src` to filter, interpret, and visualize the results.

## Environment: Google Colab

These notebooks are designed and tested to run seamlessly on **Google Colab**. This provides a free, zero-configuration environment with GPU access. Each notebook includes cells to install the necessary dependencies (`pip install ...`).

While they can be run locally, Colab is the recommended method for the best experience.

## Notebook Index

*   **`01_pytorch_detection_contract.ipynb`**: Demonstrates the "operational contract" using a pre-trained SSD (Single Shot Detector) model from **PyTorch's `torchvision`** library. This notebook shows how to apply thresholding and Non-Maximum Suppression (NMS) to the raw output of a classic object detection model.

*   **`02_tensorflow_detection_contract.ipynb`**: Shows the same "operational contract" principles applied to two different object detection models from **TensorFlow Hub**:
    1.  An SSD model trained on the COCO dataset.
    2.  An SSD model trained on the much larger Open Images V4 dataset, highlighting how the core logic remains the same even when the model and label set change.

*   **`03_yolo_detection_contract_optional.ipynb`**: Implements the contract with a modern **YOLOv8** model using the `ultralytics` library. This notebook is particularly interesting because the YOLO model performs some contract steps (like NMS) internally, providing a practical example of how our external contract can be adapted or simplified depending on the model's behavior.
