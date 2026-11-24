# Oral Disease Detection using AI

Industrial Training Project — Deep Learning / Computer Vision.
Detects oral diseases (cavities, gum disease, oral lesions) from intraoral
photos or X-rays using a transfer-learning CNN, with Grad-CAM explainability.

## Project structure

```
oral_disease_ai/
├── src/
│   ├── preprocessing.py   # dataset loading, augmentation, normalization
│   ├── model.py            # transfer-learning model (EfficientNet/ResNet)
│   ├── train.py             # full training + evaluation entry point
│   ├── gradcam.py           # Grad-CAM heatmap generation
│   └── demo_run.py          # pipeline smoke-test on synthetic data
├── scripts/
│   └── split_dataset.py    # splits a flat class-folder dataset into train/val/test
├── app/
│   └── app.py                # Streamlit prototype app (optional deliverable)
├── diagrams/                  # scripts that generate the report diagrams
├── outputs/                   # training curves, confusion matrix, Grad-CAM samples
├── models/                    # saved .keras model files
└── requirements.txt
```

## 1. Get a real dataset

This code was developed and pipeline-tested in a sandboxed environment
without internet access to Kaggle, so `outputs/` currently contains results
from a **synthetic demo run** (`src/demo_run.py`) that proves every stage of
the pipeline works end-to-end. For your actual submission, retrain on a real
dataset such as:

- Kaggle: **"Oral Diseases (panoramic x-ray)"** — salmansajid05
  `kaggle datasets download -d salmansajid05/oral-diseases`
- Kaggle: **"Oral Cancer (Lips and Tongue) Images"** — shivam17299
  `kaggle datasets download -d shivam17299/oral-cancer-lips-and-tongue-images`

(Requires a free Kaggle account + API token — see kaggle.com/docs/api.)

Run this on Google Colab (free GPU, has internet access) or your own machine.

## 2. Prepare folders

```bash
python scripts/split_dataset.py --src raw_data --out data
```

This creates `data/train`, `data/val`, `data/test`, each with one
sub-folder per class (`healthy`, `cavity`, `gum_disease`, `oral_lesion` —
rename to match whatever classes your chosen dataset provides).

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Train

```bash
python src/train.py --data_dir data --epochs 15 --backbone efficientnet
```

Produces `models/best_model.keras`, `models/final_model.keras`, and in
`outputs/`: `training_curves.png`, `confusion_matrix.png`,
`classification_report.txt`.

## 5. Explainability (Grad-CAM)

Grad-CAM is built into `src/gradcam.py` and is called automatically in the
Streamlit app; you can also call `make_gradcam_heatmap()` directly on any
image + trained model to generate a heatmap.

## 6. Run the prototype app (optional deliverable)

```bash
streamlit run app/app.py
```

Upload an image to see the predicted class and Grad-CAM overlay.

## 7. Re-run the pipeline demo (no dataset needed)

```bash
python src/demo_run.py
```

Validates the whole pipeline (data → model → train → evaluate → Grad-CAM)
on a small synthetic image set and regenerates everything in `outputs/`.

## Notes for the project report

`Oral_Disease_Detection_Report.docx` (in this same delivery) already
contains the write-up — introduction, dataset details, methodology,
evaluation criteria, results (with the generated charts embedded), and
future enhancements — matching the brief's required sections. Swap in your
real-dataset results before final submission if you retrain.
