# Multilingual-Toxic-Comment-Classification

### Project Overview:
This project aims to build a robust model for **toxic comment classification** using the **JIGSAW multilingual dataset**. The challenge involves handling the nuances of toxic language across multiple languages and developing a model that can generalize well despite being trained on English-only data.This was a competition on KAGGLE where the first place achieved Area under ROC equals to 0.9537

### Phases of the Project:
1. **Phase 1: Data Exploration and Preprocessing**
   - The dataset contains comments in multiple languages: **English (training), Spanish, Italian, Turkish (validation)**, and **Spanish, Italian, French, Portuguese, Russian, Turkish (test)**.
   - **Google Translate API** was used to translate English comments into target languages to create a multilingual dataset.
   - The evaluation metric used is the **Area Under ROC Curve (AUC)** to address class imbalance.

2. **Phase 2: Baseline Model Development**
   - A **Simple RNN** model with an embedding size of 300 was trained on tokenized text sequences.
   - Initial AUC score: **0.7896**.

3. **Phase 3: Second Model Development**
   - A more efficient model was built using **DistilBERT (distilbert-base-multilingual-cased)** for multilingual text tokenization and classification.
   - Achieved AUC score: **0.8694**.

4. **Phase 4: Final Model Development**
   - The final model used **XLM-Roberta large** with fine-tuning on the **Masked Language Modeling (MLM)** task.
   - A distributed data setup using **tf.distribute** was implemented for efficient processing on TPUs.
   - Final AUC score: **0.9414** on test data.

5. **Phase 5: Final Model Modifications**
   - Fine-tuned the **XLM-Roberta large** model on the balanced multilingual training data instead of the test data to avoid bias.
   - Final AUC score after modification: **0.9397**.

### Evaluation Summary:
| Model                         | AUC Score   |
|-------------------------------|-------------|
| **Simple RNN**                 | 0.7896      |
| **DistilBERT**                 | 0.8694      |
| **XLM-Roberta (MLM on test)**  | 0.9414      |
| **XLM-Roberta (MLM on training)** | 0.9397  |

### Key Technologies Used:
- **Keras**, **Simple RNN**
- **Hugging Face Transformers**, **DistilBERT**
- **XLM-Roberta Large** (Masked Language Model)
- **Google Translate API**
- **tf.distribute** for distributed datasets on TPUs
- **WANDB** for tracking metrics

This project showcases the challenges and solutions in building a multilingual model for toxic comment classification, leveraging cutting-edge NLP transformer models and distributed computing techniques to handle large-scale datasets efficiently.
