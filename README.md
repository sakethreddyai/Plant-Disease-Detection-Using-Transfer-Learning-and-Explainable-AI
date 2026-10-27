# Plant Disease Detection Using Transfer Learning and Explainable AI

A group machine learning project exploring maize leaf disease classification through convolutional neural networks, transfer learning, interpretable machine learning, and Explainable AI (XAI).

## Overview

Plant diseases can significantly affect crop health and agricultural productivity. This project investigates how deep learning and interpretable models can help identify maize leaf diseases while making model predictions easier to understand.

Using a publicly available maize leaf image dataset, we developed and evaluated multiple classification approaches, compared their predictive performance, investigated their decision-making processes, and explored model compression for resource-constrained deployment.

## Key Contributions

- **Comparative modelling:** Evaluated a custom Convolutional Neural Network (CNN), a fine-tuned EfficientNetB3 model, a Decision Tree, and a Random Forest.
- **Transfer learning:** Used EfficientNetB3 to extract image representations for downstream interpretable classification.
- **Explainable AI:** Applied Grad-CAM++ to visualise image regions influencing CNN predictions and investigated feature importance and decision paths in tree-based models.
- **Model compression:** Explored quantization to support efficient inference on resource-constrained devices.
- **Responsible AI considerations:** Examined potential shortcut learning, dataset bias, interpretability, and the challenges of generalising from controlled images to real-world agricultural settings.

## Dataset

The project uses a maize leaf image dataset derived from PlantVillage, containing four classes:

| Class | Description |
|---|---|
| Healthy | Healthy maize leaves |
| Common Rust | Leaves affected by common rust |
| Northern Leaf Blight | Leaves affected by northern leaf blight |
| Gray Leaf Spot | Leaves affected by gray leaf spot |

The dataset was divided into training, validation, and test sets using a 70:20:10 split. Preprocessing included image normalisation and light augmentation.

## Models and Results

The following test accuracies are reported in the project report.

| Model | Test accuracy |
|---|---:|
| Custom CNN | 81.26% |
| EfficientNetB3 | 69.93% |
| Random Forest | 63.96% |
| Decision Tree | 60.67% |

The custom CNN achieved the highest reported test accuracy among the evaluated models. The tree-based models offered more directly inspectable decision logic, illustrating the trade-off between predictive performance and interpretability.

*These figures are the reported results from the project and have not been independently reproduced.*

## Explainable AI

### Grad-CAM++

Grad-CAM++ heatmaps were used to visualise image regions influencing the CNN's predictions. The analysis examined whether the model focused on disease-related patterns, including leaf spots and discolouration, rather than irrelevant background features.

### Interpretable machine learning

Decision Trees and Random Forests were evaluated using features extracted by EfficientNetB3. Their feature importance and decision paths provided additional insight into the patterns associated with different disease classes.

The explainability analysis also highlighted the importance of investigating misclassifications and potential reliance on spurious visual cues.

## Model Efficiency

The project explored post-training quantization and TensorFlow Lite conversion as approaches to reducing model resource requirements.

These techniques are relevant to potential deployment on mobile and edge devices, where computational resources and connectivity may be limited.

## Technology Stack

- **Language:** Python
- **Deep learning:** CNNs and EfficientNetB3
- **Machine learning:** Decision Trees and Random Forests
- **Explainability:** Grad-CAM++
- **Model optimisation:** Quantization and TensorFlow Lite
- **Development:** Jupyter Notebook

## Repository Contents

- `plant_disease_detection.ipynb` — Project code, modelling experiments, and analysis.
- `REPORT.pdf` — Project report, literature review, methodology, and findings.
- `.gitignore` — Rules for excluding unnecessary local and generated files.

## Limitations and Future Work

The dataset's controlled imaging conditions may not fully represent real agricultural environments. Future work could evaluate the models on independently collected field images, investigate class-wise performance and robustness, and measure the accuracy, model size, and inference latency of compressed models.

Grad-CAM++ visualisations provide useful diagnostic evidence, but they do not establish that a model's predictions are correct or that its explanations are fully faithful.

## Academic Context

This project was completed as a group project at Technological University Dublin. The implementation and results should be interpreted in the context of the original project report.

**Author:** Saketh Reddy Chandupatla  
**Institution:** Technological University Dublin

*This project is an academic exploration of plant disease classification and explainable machine learning. It is not a validated agricultural diagnostic system.*
