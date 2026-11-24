const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, ImageRun, BorderStyle, PageBreak,
} = require("docx");

const OUT = "outputs";

const IMG_DIMS = {
  "outputs/pipeline_diagram.png": [1612, 87],
  "outputs/architecture_diagram.png": [265, 597],
  "outputs/training_curves.png": [1650, 600],
  "outputs/confusion_matrix.png": [900, 750],
  "outputs/gradcam_samples.png": [1950, 975],
  "outputs/sample_predictions.png": [1800, 1200],
};

function img(path, width) {
  const buf = fs.readFileSync(path);
  const [w, h] = IMG_DIMS[path];
  const ratio = h / w;
  return new ImageRun({
    data: buf,
    transformation: { width, height: Math.round(width * ratio) },
    type: path.endsWith(".png") ? "png" : "jpg",
  });
}

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun(text)], spacing: { after: 120 }, ...opts });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20, color: "555555" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
  });
}
function centerImg(path, width, captionText) {
  return [
    new Paragraph({ children: [img(path, width)], alignment: AlignmentType.CENTER, spacing: { after: 60 } }),
    caption(captionText),
  ];
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2500, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1F4E78" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000" })],
    })],
  });
}

function simpleTable(headers, rows, widths) {
  return new Table({
    width: { size: 9500, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ children: headers.map((h, i) => cell(h, { header: true, width: widths[i] })) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, { width: widths[i] })) })),
    ],
  });
}

const classificationReport = fs.readFileSync(`${OUT}/classification_report.txt`, "utf-8");

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } }, // US Letter
    children: [
      // Title page
      new Paragraph({ text: "", spacing: { before: 1200 } }),
      new Paragraph({
        children: [new TextRun({ text: "Oral Disease Detection using AI", bold: true, size: 44 })],
        alignment: AlignmentType.CENTER, spacing: { after: 200 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Deep Learning Project Report", size: 28, color: "555555" })],
        alignment: AlignmentType.CENTER, spacing: { after: 400 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Industrial Training Program on Artificial Intelligence & Machine Learning", size: 22, italics: true })],
        alignment: AlignmentType.CENTER, spacing: { after: 1200 },
      }),
      new Paragraph({ children: [new PageBreak()] }),

      // 1. Introduction
      h1("1. Introduction"),
      p("Oral diseases such as dental caries (cavities), periodontal (gum) disease, and oral mucosal lesions affect a very large share of the global population, and are most treatable when caught early. Access to a dentist for routine screening is not available to everyone, especially in rural and low-resource settings, which delays diagnosis and treatment."),
      p("This project builds an AI-based system that classifies intraoral photographs or dental X-rays into diagnostic categories (healthy, cavity, gum disease, oral lesion) using a convolutional neural network built on top of a pretrained backbone (transfer learning). Alongside the classification, the system generates a Grad-CAM heatmap that highlights the exact image region driving each prediction, so a dentist or health worker can see — not just be told — why the model reached its conclusion."),

      // 2. Objectives
      h1("2. Project Objectives"),
      bullet("Collect and preprocess oral disease image datasets (X-rays and intraoral photographs)."),
      bullet("Train a deep learning model using CNN-based transfer learning (EfficientNet / ResNet)."),
      bullet("Build a classifier that differentiates healthy vs. diseased oral conditions across multiple categories."),
      bullet("Provide visual Grad-CAM heatmaps that explain each prediction."),
      bullet("Integrate the trained model into a simple interface for real-time detection."),

      // 3. Dataset
      h1("3. Dataset Details"),
      p("The pipeline is designed to work with open-source oral/dental image datasets, including:"),
      bullet('"Oral Diseases (panoramic X-ray)" dataset — Kaggle, by S. Sajid — panoramic dental X-rays labeled by condition.'),
      bullet('"Oral Cancer (Lips and Tongue) Images" dataset — Kaggle, by S. Barot & P. Patel — intraoral photographs for oral cancer indicators.'),
      p("Both are freely available on Kaggle and were selected because they match the four target categories in this project's scope (healthy, cavity, gum disease, oral lesion / cancer indicator). Images are organized by class into folders, then split 70% / 15% / 15% into train, validation, and test sets (scripts/split_dataset.py automates this split)."),
      p("Note on this report's sample results: this development environment does not have direct internet access to Kaggle, so the results shown in Section 6 come from a small representative synthetic image set used to validate that every stage of the pipeline (data loading → model → training → evaluation → Grad-CAM) runs correctly end-to-end. Before final submission, re-run src/train.py on the real dataset (instructions in README.md) and swap in the resulting charts — the code itself does not change.", { }),

      // 4. Methodology
      h1("4. Methodology"),
      h2("4.1 System Pipeline"),
      ...centerImg(`${OUT}/pipeline_diagram.png`, 560, "Figure 1: End-to-end system pipeline"),

      h2("4.2 Dataset Preparation"),
      p("Images are resized to 224x224, normalized to [0,1], and augmented during training only (random horizontal flip, small rotation, zoom, and contrast jitter) to reduce overfitting given the moderate dataset size typical of medical imaging tasks."),

      h2("4.3 Model Selection & Architecture"),
      p("The classifier uses transfer learning: a CNN backbone (EfficientNetB0 or ResNet50) pretrained on ImageNet is used as a frozen feature extractor, topped with a global average pooling layer, dropout, a dense layer, and a softmax output layer over the four classes. After the head converges, the top backbone layers are unfrozen for a short fine-tuning phase at a lower learning rate."),
      ...centerImg(`${OUT}/architecture_diagram.png`, 320, "Figure 2: Model architecture"),

      h2("4.4 Training & Testing"),
      p("The dataset is split into train/validation/test sets. The model is trained with the Adam optimizer and categorical cross-entropy loss, with early stopping and checkpointing on validation accuracy. A short fine-tuning phase follows initial training."),

      h2("4.5 Explainability (Grad-CAM)"),
      p("Grad-CAM computes gradients of the predicted class with respect to the last convolutional layer's feature maps, producing a heatmap that is overlaid on the original image to show which region most influenced the prediction — important for building clinical trust in the model's output."),

      h2("4.6 Interface Development"),
      p("A lightweight Streamlit app (app/app.py) lets a user upload an image and instantly see the predicted class alongside its Grad-CAM overlay, fulfilling the optional prototype-app deliverable."),

      // 5. Evaluation criteria
      h1("5. Evaluation Criteria Addressed"),
      simpleTable(
        ["Criterion", "Weight", "How this project addresses it"],
        [
          ["Functionality", "40%", "End-to-end trained classifier with measured accuracy, precision, recall, F1 (Section 6)."],
          ["Creativity", "30%", "Combines classification with Grad-CAM explainability for clinical interpretability."],
          ["Technical Accuracy", "20%", "Proper CV/ML pipeline: augmentation, transfer learning, fine-tuning, Grad-CAM, evaluation metrics."],
          ["Documentation", "10%", "This report plus inline code documentation and README with reproduction steps."],
        ],
        [2600, 1400, 5500]
      ),

      // 6. Results
      new Paragraph({ children: [new PageBreak()] }),
      h1("6. Results"),
      p("The following results are from the pipeline-validation demo run described in Section 3. Re-run on the real dataset before final submission to replace these with clinical results."),

      h2("6.1 Training Curves"),
      ...centerImg(`${OUT}/training_curves.png`, 560, "Figure 3: Training and validation accuracy/loss over epochs"),

      h2("6.2 Confusion Matrix"),
      ...centerImg(`${OUT}/confusion_matrix.png`, 380, "Figure 4: Confusion matrix on the held-out test set"),

      h2("6.3 Classification Report"),
      ...classificationReport.split("\n").map(line =>
        new Paragraph({
          children: [new TextRun({ text: line || " ", font: "Courier New", size: 18 })],
          spacing: { after: 0 },
        })
      ),
      new Paragraph({ text: "", spacing: { after: 200 } }),

      h2("6.4 Grad-CAM Sample Outputs"),
      ...centerImg(`${OUT}/gradcam_samples.png`, 560, "Figure 5: Original images (top) vs. Grad-CAM heatmaps (bottom) for one sample per class"),

      h2("6.5 Sample Prediction Outputs"),
      ...centerImg(`${OUT}/sample_predictions.png`, 480, "Figure 6: Random test-set predictions (green = correct, red = incorrect)"),

      // 7. Applications
      h1("7. Applications"),
      bullet("Early detection of oral diseases in rural and remote areas without easy dental access."),
      bullet("Assisting dentists with faster, explainable preliminary diagnosis."),
      bullet("Public health screening campaigns for oral hygiene awareness."),
      bullet("Low-cost AI-powered oral healthcare triage tool for developing regions."),

      // 8. Future enhancements
      h1("8. Future Enhancements"),
      bullet("Expand the dataset with multilingual/global data for better generalization across populations."),
      bullet("Integrate voice-based results explanation for accessibility."),
      bullet("Deploy on smartphones for offline use in rural healthcare settings."),
      bullet("Extend to multi-disease detection covering oral, throat, and dental issues together."),

      // 9. Conclusion
      h1("9. Conclusion"),
      p("This project demonstrates a complete, reproducible pipeline for AI-assisted oral disease detection: dataset preparation, transfer-learning-based classification, Grad-CAM explainability, quantitative evaluation, and an interactive prototype interface. The code is modular (src/preprocessing.py, model.py, train.py, gradcam.py) so it can be retrained directly on any of the referenced open datasets to produce clinically meaningful results."),

      // References
      h1("References"),
      bullet("Sajid, S. Oral Diseases (panoramic X-ray) dataset. Kaggle, 2023."),
      bullet("Barot, S. & Patel, P. Oral Cancer (Lips and Tongue) Images. Kaggle, 2020."),
      bullet("Tan, M. & Le, Q. EfficientNet: Rethinking Model Scaling for CNNs. ICML, 2019."),
      bullet("He, K. et al. Deep Residual Learning for Image Recognition (ResNet). CVPR, 2016."),
      bullet("Selvaraju, R. R. et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV, 2017."),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(`${OUT}/Oral_Disease_Detection_Report.docx`, buf);
  console.log("Report written.");
});
