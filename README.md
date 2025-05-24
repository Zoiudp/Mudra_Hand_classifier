# Hand_classifier
 Classificador de mudras por dados dos controles VR

## Installation

To install all the required libraries, simply run:

```bash
pip install -r requirements.txt
```

## Usage

To run the hand classifier and see the model perform 10,000 predictions on the dataset, execute:

```bash
python Hand_classifier.py
```

This will process the `combined_one_hand_data_with_classification.csv` file and output prediction results.

## Checking Model Accuracy

To examine the model's accuracy and other performance metrics:

1. Open `classifier.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells in the notebook
3. Review the accuracy scores, confusion matrix, and other evaluation metrics

```bash
jupyter notebook classifier.ipynb
# or
jupyter lab
```

The notebook contains detailed analysis of different model performances on the hand gesture (mudra) classification task.