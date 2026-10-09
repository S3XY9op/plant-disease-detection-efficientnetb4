# Plant Disease Detection Using EfficientNetB4

A deep-learning project for classifying plant leaf images into disease/healthy categories using transfer learning with **TensorFlow/Keras EfficientNetB4**. The notebook trains a classifier, fine-tunes the pretrained backbone, evaluates it on a held-out test set, and explores a confusion matrix and class-wise ROC/AUC.

> **Reported notebook result:** test accuracy of approximately **99.21%** (`0.9921`) in the saved notebook output. This is the result recorded by that run, not a guarantee of performance on new images. Re-run the notebook to verify results in your environment.

## Project contents

- `plant_disease_detection.ipynb` — training and evaluation notebook.
- `requirements.txt` — Python packages used by the notebook.
- `.gitignore` — excludes generated datasets, model files, and local environments.
- `CITATION.cff` — citation metadata template; update the author details before publishing if desired.

## Dataset

The notebook references a Plant Leave Diseases dataset hosted on Mendeley Data:

- Dataset landing page: https://data.mendeley.com/datasets/tywbtsjrjv/1
- The notebook's download cell retrieves a ZIP archive of roughly 905 MB. Check the dataset page for its current terms, version, and attribution requirements.

**Do not commit the dataset ZIP or extracted images to GitHub.** Keep large data files out of the repository and download them locally/in Colab as needed. The notebook currently uses Google Colab paths such as `/content/`; see the setup notes below.

## Run in Google Colab

1. Upload `plant_disease_detection.ipynb` to Google Colab, or open it from your GitHub repository.
2. Select **Runtime → Change runtime type → T4 GPU** (GPU is optional but recommended).
3. Run the dataset-download cell before the extraction cell. The notebook currently has an extraction cell before the download cell, so move the `wget` download cell above `!unzip /content/dataset.zip` or run the download cell first.
4. Ensure the extracted dataset folder matches the path expected by the notebook: `Plant_leave_diseases_dataset_with_augmentation`.
5. Run the notebook from top to bottom. It splits data into training, validation, and test sets, trains the model, fine-tunes it, and evaluates the result.

## Run locally

Python 3.10 or 3.11 is a reasonable starting point for the listed TensorFlow dependencies. Install the packages:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The notebook's `/content/...` paths and shell commands are Colab-oriented. For local execution, update those paths and download/extract the dataset to the directory your notebook expects. A GPU is optional; training may be much slower on CPU.

## Method overview

1. Load and split images into train/validation/test sets.
2. Resize images to `160 × 160` and batch them.
3. Use ImageNet-pretrained EfficientNetB4 as a feature extractor.
4. Add global average pooling, dropout, and a softmax classification layer.
5. Train the classifier, then unfreeze part of the backbone for fine-tuning.
6. Evaluate test accuracy/loss and inspect confusion-matrix and ROC/AUC results.

## Important reproducibility notes

- The notebook uses a dataset split seed (`1337`), but other sources of randomness may still affect results.
- The notebook's final ROC/AUC cell hard-codes a class-name list. Confirm it matches `train_dataset.class_names` in the exact dataset version used.
- Review notebook execution order and outputs before presenting results as final research findings.
- This is an experimental image-classification project, not a substitute for expert agricultural diagnosis.

## Citation and licensing

Please cite the dataset according to its Mendeley Data record and comply with its license/terms. No project license is assigned here because the appropriate license depends on your intended reuse rights. If you want others to reuse your code, choose a license and add a `LICENSE` file before publishing.

## Suggested repository description

Plant leaf disease classification using TensorFlow/Keras and transfer learning with EfficientNetB4, including fine-tuning and evaluation with test accuracy, confusion matrix, and ROC/AUC analysis.
