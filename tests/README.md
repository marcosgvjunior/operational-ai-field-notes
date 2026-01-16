# Testing Philosophy & Strategy

This directory contains the tests that ensure the correctness and reliability of the project's codebase. Our testing strategy is modeled after the classic testing pyramid, which emphasizes a multi-layered approach to quality assurance.

## The Testing Pyramid

A complete testing strategy for an ML project involves several layers. Our goal is to progressively build out this pyramid:

1.  **Unit Tests (Currently Implemented):** These form the foundation of the pyramid. They are fast, numerous, and isolated. Their purpose is to verify the correctness of individual, self-contained pieces of logic.
2.  **Integration Tests (Future Goal):** These tests sit in the middle layer. They verify that different components of the system work together correctly. For this project, this means testing the connection between our core `src` logic and the specific ML framework adapters.
3.  **End-to-End (E2E) Tests (Future Goal):** These tests are at the top of the pyramid. They validate the entire workflow from start to finish, just as a user would experience it. In our case, this would involve running the Jupyter notebooks to ensure they execute without error and produce the expected outputs.

## Current Implementation: Unit Tests

**At present, this directory contains the first and most critical layer: Unit Tests.**

*   **Scope:** The tests exclusively target the framework-agnostic primitives defined in the `../src/` directory. They validate the mathematical and logical correctness of core "operational contract" components:
    *   Intersection over Union (IoU) invariants
    *   Non-Maximum Suppression (NMS) filtering behavior
    *   Bounding box transformations and agreements

*   **Technology:** The suite is built with `pytest`. Tests use simple, hard-coded `numpy` arrays as inputs to ensure they are fully deterministic and repeatable.

*   **Philosophy:** These tests are intentionally isolated from heavy external dependencies. **Running them does not require PyTorch, TensorFlow, or any other ML framework.** This makes the test suite extremely fast and guarantees that we are testing *our* logic, not the behavior of a third-party library.

## How to Run the Tests

To run the full unit test suite, ensure you have the dependencies from `requirements.txt` installed. Then, from the project's root directory, execute `pytest`.

```bash
# From the project root
pytest
```

## Future Work: Expanding Test Coverage

As the project grows, our goal is to build out the upper layers of the testing pyramid:
*   **Integration Tests:** We plan to add tests that load actual framework adapters (e.g., from `src/tfhub_det_openimages.py`) and verify they correctly process inputs and parse model outputs.
*   **Notebook Regression Tests:** We will implement an E2E testing strategy to automatically run the Jupyter notebooks in `../notebooks/`. This will ensure that changes to the core logic do not break the examples and that the primary deliverables of the project remain functional.
