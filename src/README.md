# Source Code (`src`)

This directory contains the core, framework-agnostic Python modules that implement the project's "operational contract." The code here is engineered not just to work, but to serve as a clear, maintainable, and robust reference implementation of key software design principles applied to an ML context.

## Guiding Engineering Principles

The design of this codebase is a deliberate exercise in software craftsmanship. The following principles are not just guidelines; they are strictly enforced.

*   **SOLID Principles as a Foundation:** The code is heavily influenced by the SOLID design principles, ensuring a scalable and maintainable architecture.
    *   **Single Responsibility Principle (SRP):** Each module and function has one, and only one, reason to change. For example, `boxes.py` is exclusively concerned with the geometry and logic of bounding boxes (IoU, NMS), while `viz.py` is exclusively concerned with visualization.
    *   **Low Coupling:** The core logic is completely decoupled from any specific ML framework. Modules like `boxes.py` do not `import torch` or `import tensorflow`. This ensures the core algorithms can be used anywhere and are immune to changes in framework APIs.
    *   **High Cohesion:** Functions within a module are strongly related and focused on a single area of responsibility. This makes the codebase easier to navigate and understand.

*   **Code Quality and Metrics:** We strive for objectively high-quality code.
    *   **Low Cyclomatic Complexity:** Functions are intentionally kept short and linear. Complex branching, nested loops, and deep `if/else` chains are avoided. This makes the code easier to read, test, and reason about, drastically reducing the risk of hidden bugs.
    *   **No Hidden State:** All functions are explicit about their inputs and outputs. There is no hidden global state, making the system's behavior predictable and deterministic.

*   **Documentation and Style:** Code is written for human understanding first.
    *   **PEP 8 Compliance:** The entire codebase adheres to the PEP 8 style guide for Python code, ensuring a high level of consistency and readability.
    *   **reStructuredText (reST) Docstrings:** All public modules and functions are documented with comprehensive docstrings using the reST format. This is the standard used by many Python projects and tools like Sphinx.
    *   **Ready for Sphinx:** The detailed docstrings are written with the intention of being automatically parsed by **Sphinx** in the future to generate a full, professional documentation website.

*   **Testability:** The system is designed for correctness, which is verified through testing.
    *   By decoupling the core logic from the ML frameworks, we can write precise and reliable unit tests for every component of the "operational contract." The full test suite resides in the `../tests/` directory and serves as live documentation for the expected behavior of the code.

## Modules

*   `boxes.py`: Defines the primary `Box` data structure and the core "operational contract" functions, including Intersection over Union (`iou`) and Non-Maximum Suppression (`nms`).
*   `coco_labels.py`: Provides a utility mapping from MS COCO dataset class indices to human-readable string labels.
*   `tfhub_det_openimages.py`: An adapter module containing a wrapper for a specific TensorFlow Hub object detection model (SSD w/ MobileNetV2) trained on the Open Images V4 dataset.
*   `unet_segmentation_stub.py`: A placeholder file for future work on image segmentation, aligning the repository with the project roadmap.
*   `viz.py`: Contains utility functions for drawing bounding boxes, labels, and scores on images to visualize model outputs.
