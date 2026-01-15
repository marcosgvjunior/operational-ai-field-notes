# notebooks/

Three notebooks, same contract, different toolchains:

- `01_pytorch_detection_contract.ipynb`
  - torchvision SSD + MobileNet (detection)

- `02_tensorflow_detection_contract.ipynb`
  - TF Hub SSD + MobileNet (detection)

- `03_yolo_detection_contract_optional.ipynb`
  - ultralytics YOLO (detection), optional because it adds a heavier dependency

All notebooks then use the same contract primitives from `src/`:
threshold -> IoU -> NMS

Images:
- `data/input/images/image1.png`
- `data/input/images/image2.png`
