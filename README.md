# MNIST Handwritten Digit Classification

## Overview

This project develops an end-to-end multiclass Machine Learning pipeline for handwritten digit classification using the MNIST dataset.

## Project Objective

The objective is to compare three different Machine Learning approaches, evaluate their generalization performance, investigate Out-of-Distribution behavior, and apply the best model to external handwritten digits.

## Dataset

MNIST contains 70,000 grayscale handwritten digit images. Each image has 28 × 28 pixels and is represented by 784 numerical features.

## Project Workflow

1. Dataset loading
2. Exploratory Data Analysis
3. Stratified train-validation-test splitting
4. Pixel normalization
5. Hyperparameter tuning
6. Final model training
7. Independent test evaluation
8. Confusion-matrix and error analysis
9. Model comparison
10. Hidden-class and OOD experiment
11. Custom handwritten digit preprocessing and classification

## Machine Learning Models

- K-Nearest Neighbors
- Random Forest
- Multilayer Perceptron

## Hyperparameter Tuning

At least two hyperparameters were investigated for every model using the validation set.

### KNN
- `n_neighbors`
- `weights`

### Random Forest
- `n_estimators`
- `max_depth`

### MLP
- `hidden_layer_sizes`
- `learning_rate_init`

## Evaluation Metrics

The final models were compared using:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Training time
- Inference time
- 10 × 10 confusion matrices

## Model Comparison

| model         |   accuracy |   weighted_precision |   weighted_recall |   weighted_f1 |   training_time_seconds |   inference_time_seconds |
|:--------------|-----------:|---------------------:|------------------:|--------------:|------------------------:|-------------------------:|
| KNN           |   0.973214 |             0.973408 |          0.973214 |      0.973185 |               0.0420513 |                 14.7071  |
| MLP           |   0.973071 |             0.973105 |          0.973071 |      0.973047 |              19.1358    |                  0.02445 |
| Random Forest |   0.968143 |             0.968142 |          0.968143 |      0.968122 |              20.0386    |                  0.39434 |

## Best Model

The selected model was **KNN** with a weighted F1-score of **0.9732** and Accuracy of **0.9732**.

## Out-of-Distribution Experiment

Digits [4, 9] were removed from the masked model's training data and later used exclusively as OOD test samples.

The masked model produced a mean maximum OOD confidence of **0.8725**. This demonstrates that a closed-set classifier can generate confident predictions for classes that were never observed during training.

## Custom Handwritten Digit Classification

External handwritten digits are placed in:

`data/custom_digits/`

The preprocessing pipeline performs grayscale conversion, conditional inversion, bounding-box detection, aspect-ratio-preserving resize, centering, normalization, and conversion to a 784-feature vector.

Current execution status: 9 custom handwritten image(s) were processed.

## Project Structure

```text
mnist-classification-project/
├── mnist_classification_project.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── custom_digits/
├── images/
│   ├── dataset_samples/
│   ├── confusion_matrices/
│   ├── model_comparison/
│   ├── ood_analysis/
│   └── custom_predictions/
└── docs/
    ├── project_notes.md
    └── video_script.md
```

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies.
4. Start Jupyter Notebook or JupyterLab.
5. Open `mnist_classification_project.ipynb`.
6. Run the notebook from top to bottom.

## Key Findings

- Best model: **KNN**
- Accuracy: **0.9732**
- Weighted F1-score: **0.9732**
- OOD mean maximum confidence: **0.8725**

## Limitations

- MNIST is cleaner than real-world handwriting.
- Standard tabular classifiers do not explicitly exploit spatial image structure.
- OOD confidence is not equivalent to calibrated uncertainty.
- Runtime measurements depend on hardware.

## Future Improvements

- Convolutional Neural Networks
- Data augmentation
- Probability calibration
- Explicit OOD detection
- Robustness tests under image noise and transformations

## Technologies

Python, NumPy, Pandas, Matplotlib, Scikit-learn, Pillow, OpenCV, Jupyter Notebook, Tabulate.

## Author

Renan de Brito Leme\n