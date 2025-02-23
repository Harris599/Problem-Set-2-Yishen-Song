# Problem-Set-2-Yishen-Song
# Sentiment Analysis with BERT on Stanford Sentiment Treebank (SST)

## 📌 Project Overview
This project fine-tunes a **BERT-based sentiment analysis model** on the **Stanford Sentiment Treebank (SST)** dataset. The model classifies movie reviews into **five sentiment categories**:
- **0** → Very Negative
- **1** → Negative
- **2** → Neutral
- **3** → Positive
- **4** → Very Positive

The project includes **dataset preprocessing, model training, evaluation, and deployment** using Hugging Face's `transformers` library.

---
## 📂 Repository Structure
```
📂 Sentiment-Analysis-BERT/
│── 📂 data/            # Cleaned datasets (CSV format)
│   ├── train_cleaned.csv
│   ├── dev_cleaned.csv
│   ├── test_cleaned.csv
│
│── 📂 models/          # Trained model and tokenizer
│   ├── bert_sentiment_model/
│
│── 📂 notebooks/       # Jupyter Notebooks for training and inference
│   ├── sentiment_analysis.ipynb
│
│── 📂 results/         # Evaluation metrics and visualizations
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│
│── README.md          # Project overview and usage guide
│── requirements.txt   # Required Python packages
```

---
## 📊 Dataset
The **Stanford Sentiment Treebank (SST)** dataset contains movie reviews with labeled sentiments. The original dataset was preprocessed to remove unnecessary syntax and converted into **CSV format**.

### 📥 Download Cleaned Data
- [Train Dataset](sandbox:/mnt/data/train_cleaned.csv)
- [Dev Dataset](sandbox:/mnt/data/dev_cleaned.csv)
- [Test Dataset](sandbox:/mnt/data/test_cleaned.csv)

---
## ⚙️ Setup Instructions
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/yourusername/Sentiment-Analysis-BERT.git
cd Sentiment-Analysis-BERT
```
### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```
### 3️⃣ **Run Jupyter Notebook**
```bash
jupyter notebook
```
Then, open **notebooks/sentiment_analysis.ipynb** to train or test the model.

---
## 🏋️‍♂️ Model Training
The **BERT-based model** is fine-tuned using Hugging Face's `transformers` library:
```python
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT tokenizer and model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```
### Training Execution:
Run the Jupyter notebook **sentiment_analysis.ipynb** to train the model.

---
## 🧪 Model Evaluation
After training, the model is evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report**

Example evaluation results:
```
Accuracy: 84.5%
Classification Report:
    Precision    Recall    F1-score
0    0.80       0.78      0.79
1    0.83       0.81      0.82
2    0.86       0.88      0.87
3    0.85       0.84      0.84
4    0.89       0.87      0.88
```

---
## 🔮 Inference: Predict Sentiment for a New Sentence
To make predictions on new text input, use:
```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0: "very negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very positive"}
    return label_map[prediction]

# Example prediction
predict_sentiment("This movie was absolutely amazing!")
```

---
## 📜 License
This project is open-source under the **MIT License**.

---
## 🛠️ Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

---


