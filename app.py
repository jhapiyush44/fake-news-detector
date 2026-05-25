import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="hamzab/roberta-fake-news-classification"
)

def predict_news(text):

    if len(text.strip()) == 0:
        return "Please enter some text."

    result = classifier(text)[0]

    label = result["label"]
    score = result["score"] * 100

    if label == "LABEL_1":
        prediction = "🟢 REAL NEWS"
    else:
        prediction = "🔴 FAKE NEWS"

    return f"""
{prediction}

Confidence Score: {score:.2f}%
"""

demo = gr.Interface(
    fn=predict_news,

    inputs=gr.Textbox(
        lines=12,
        placeholder="Paste news article here..."
    ),

    outputs=gr.Textbox(label="Prediction"),

    title="📰 AI Fake News Detector",

    description="Fake News Detection using Hugging Face Transformers",

    theme="soft"
)

demo.launch()