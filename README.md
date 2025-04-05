# Audio Grammar Scoring Model

## Overview

This project aims to predict the grammar quality score (from 0 to 5) of spoken English audio samples. It was developed for the SHL Hiring Assessment.

The code takes `.wav` audio files as input and uses machine learning to output a grammar score for each file.

## Data Requirements

The script expects the data to be organized as follows (like in the Kaggle competition input):

*   `/kaggle/input/shl-audio/dataset/audios_train/`: Folder containing the training `.wav` audio files.
*   `/kaggle/input/shl-audio/dataset/audios_test/`: Folder containing the test `.wav` audio files.
*   `/kaggle/input/shl-audio/dataset/train.csv`: A CSV file listing training audio filenames and their corresponding grammar scores (labels).
*   `/kaggle/input/shl-audio/dataset/test.csv`: A CSV file listing test audio filenames (the labels in this file are ignored).
*   `/kaggle/input/shl-audio/dataset/sample_submission.csv`: A CSV file showing the required output format and the correct order for test predictions.

## How it Works (Simplified Steps)

1.  **Load Data Info:** Reads the list of training files and their scores (`train.csv`) and the required test file order (`sample_submission.csv`).
2.  **Extract Features:** For each audio file:
    *   Loads the audio using the `librosa` library.
    *   Calculates several audio features, focusing on the **average (mean)** and **variation (standard deviation)** of:
        *   MFCCs (Mel-Frequency Cepstral Coefficients)
        *   Chroma Features
        *   Spectral Contrast
        *   Zero Crossing Rate
        *   RMS Energy
    *   Combines these numbers into a single list (feature vector) for each audio file.
3.  **Scale Features:** Adjusts the range of all feature values using `StandardScaler` so that they have a similar scale. This helps the machine learning model perform better. The scaler is set up (fitted) using only the *training* data.
4.  **Train Model:** Trains an `XGBoost Regressor` model using the scaled features and known scores from the *training* data. XGBoost is good at finding complex patterns.
5.  **Process Test Data:**
    *   Extracts the same features (mean and std dev) for each *test* audio file, making sure to process them in the order required by `sample_submission.csv`.
    *   Scales these test features using the *same* scaler that was set up with the training data.
6.  **Predict Scores:** Uses the trained XGBoost model to predict grammar scores for the scaled test features.
7.  **Create Submission File:**
    *   Makes sure the predicted scores are within the valid range (0 to 5).
    *   Saves the predictions into a file named `submission.csv` with two columns: `filename` and `label`.

## Requirements

You need Python 3 and the following libraries:

*   `numpy`
*   `pandas`
*   `librosa`
*   `scikit-learn`
*   `xgboost`
*   `tqdm` (for progress bars)

You can usually install them using pip:
`pip install numpy pandas librosa scikit-learn xgboost tqdm`

## How to Run

1.  Make sure you have the dataset available in the `/kaggle/input/shl-audio/dataset/` directory (or update the paths near the top of the script if your data is elsewhere).
2.  Ensure all required libraries are installed.
3.  Run the Python script (e.g., `python your_script_name.py` or run the cell in a Kaggle/Jupyter notebook).

## Output

The script will create a file named `submission.csv` in the same directory where it's run. This file contains the predicted grammar scores for the test audio files, ready for submission to the competition.
