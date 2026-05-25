---
title: AI Fake News Detector
emoji: 📰
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
short_description: Fake News Detection using Hugging Face Transformers and Gradio
---

# 📰 AI Fake News Detector

An AI-powered Fake News Detection application built using Hugging Face Transformers and Gradio.

This application analyzes news articles or headlines and predicts whether the news is **REAL** or **FAKE** using a Transformer-based Natural Language Processing (NLP) model.

---

# 🚀 Live Demo

Try the deployed application directly on Hugging Face Spaces.

The model takes user input in the form of:
- News headlines
- News articles
- Social media news text

and predicts:
- 🟢 REAL NEWS
- 🔴 FAKE NEWS

along with a confidence score.

---

# ✨ Features

- Real-time fake news detection
- Interactive Gradio web interface
- Transformer-based NLP pipeline
- Confidence score prediction
- Hugging Face Spaces deployment
- Lightweight and fast inference

---

# 🧠 Model Used

This project uses the Hugging Face Transformer model:

`hamzab/roberta-fake-news-classification`

Built using:
- RoBERTa Transformer Architecture
- Hugging Face Transformers
- PyTorch Backend

---

# 🛠️ Tech Stack

## Languages & Frameworks
- Python
- Gradio
- Hugging Face Transformers
- PyTorch

## Deployment
- Hugging Face Spaces

---

# 📂 Project Structure

```bash
fake-news-detector/
│
├── app.py
├── train.py
├── requirements.txt
├── runtime.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/jhapiyush44/fake-news-detector.git
cd fake-news-detector
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

## Windows

```bash
venv\Scripts\activate
```

## Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the application:

```bash
python app.py
```

The Gradio application will launch locally.

---

# 📊 Example Predictions

## Example 1

### Input
```text
Scientists confirm aliens built the pyramids.
```

### Output
```text
🔴 FAKE NEWS
Confidence Score: 98.21%
```

---

## Example 2

### Input
```text
NASA announces new lunar exploration mission for 2027.
```

### Output
```text
🟢 REAL NEWS
Confidence Score: 95.67%
```

---

# 📈 Future Improvements

- Fine-tune custom transformer model
- Add multilingual fake news detection
- Add explainable AI visualizations
- Add news source credibility scoring
- Deploy with GPU acceleration
- Add batch prediction support

---

# 🎯 Learning Outcomes

This project demonstrates practical knowledge of:

- Natural Language Processing (NLP)
- Transformer Models
- Hugging Face Ecosystem
- Model Inference Pipelines
- AI Application Deployment
- Gradio Interface Development

---

# 👨‍💻 Author

Piyush Jha

BS in Data Science and Applications — IIT Madras

Interested in:
- Generative AI
- LLM Systems
- NLP
- MLOps
- AI Engineering

---

# 📜 License

This project is licensed under the MIT License.