## Emotion Model

We adopt Russell's Circumplex Model of Affect and classify emotions into four quadrants:

| Class | Emotion | Valence  | Arousal |
| ----- | ------- | -------- | ------- |
| 0     | Happy   | Positive | High    |
| 1     | Sad     | Negative | Low     |
| 2     | Angry   | Negative | High    |
| 3     | Relaxed | Positive | Low     |

Ground-truth labels are generated from participants' Self-Assessment Manikin (SAM) ratings.

---

## Data Collection

### Experimental Setup

For each video trial:

1. 60-second baseline recording
2. 120-second emotion-inducing video clip
3. SAM self-assessment
4. 20-second washout period

### Modalities

#### Facial Modality

RGB video is processed using OpenFace to extract:

* Facial Action Units (AUs)
* Gaze direction
* Head pose features

#### Physiological Modality

PPG signals are collected at:

* 250 Hz sampling rate

HeartPy is used to derive physiological features including:

* Heart Rate (BPM)
* SDNN
* RMSSD
* PNN20 / PNN50
* Breathing Rate
* Poincaré Plot Features

---

## Data Preprocessing

### PPG Pipeline

The raw PPG signal contains clipping artefacts and noise.

Processing steps:

* Initial signal trimming
* Frequency estimation using periodogram
* Adaptive Butterworth band-pass filtering
* Peak detection with HeartPy
* Feature extraction

### Facial Feature Processing

OpenFace outputs are filtered using:

* success == 1
* confidence ≥ 0.70

For each feature:

* Baseline correction
* Temporal segmentation
* Window-level aggregation

### Temporal Alignment

Both modalities are synchronised and segmented into:

* Six non-overlapping 20-second windows per trial

This produces multimodal samples suitable for downstream fusion models.

---

## Fusion Strategies

### 1. Early Fusion

Feature-level fusion through direct concatenation:

OpenFace Features + PPG Features
→ Normalisation
→ Mutual Information Feature Selection
→ Classifier

Classifiers:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest

---

### 2. Late Fusion

Decision-level fusion using modality-specific models:

OpenFace Model
PPG Model
→ Probability Outputs
→ Weighted Soft Voting
→ Final Prediction

Fusion weights are estimated dynamically using training-fold balanced accuracy.

---

### 3. Model-Level Fusion

Deep multimodal representation learning:

Facial Branch:

* Frame-wise CNN
* Temporal pooling

PPG Branch:

* 1D CNN

Fusion:

* Feature concatenation
* Fully connected layers
* Dropout regularisation

The learned embeddings are evaluated using classical downstream classifiers.

---

## Evaluation

### Validation Protocol

Leave-One-Video-Out Cross Validation (LOVO)

For each fold:

* One complete video clip is held out for testing
* Remaining clips are used for training
* Validation clips are selected only from the training partition

This prevents temporal leakage between train and test samples.

### Metrics

* Accuracy
* Balanced Accuracy
* Macro F1 Score
* Precision
* Recall
* Confusion Matrix

Evaluation is performed at:

* Window Level
* Video Level

---

## Repository Structure

```text
project/
│
├── data/
│   ├── ppg/
│   └── openface/
│
├── preprocessing/
│   ├── ppg_processing.py
│   ├── openface_processing.py
│   └── synchronization.py
│
├── models/
│   ├── early_fusion/
│   ├── late_fusion/
│   └── model_level_fusion/
│
├── evaluation/
│   ├── metrics.py
│   └── lovo_validation.py
│
├── notebooks/
│
└── README.md

# ACHRI
