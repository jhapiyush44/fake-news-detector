import gradio as gr
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

model_path = "model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()


def predict_news(text):

    if len(text.strip()) == 0:
        return "Please enter some news text."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(
            outputs.logits,
            dim=-1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    prediction = prediction.item()
    confidence = confidence.item() * 100

    if prediction == 1:
        label = "🟢 REAL NEWS"
    else:
        label = "🔴 FAKE NEWS"

    return f"{label}\n\nConfidence Score: {confidence:.2f}%"


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 📰 AI Fake News Detector

        Detect whether a news article is REAL or FAKE using a fine-tuned DistilBERT model.
        """
    )

    news_input = gr.Textbox(
        lines=12,
        placeholder="Paste your news article here...",
        label="News Article"
    )

    output = gr.Textbox(
        label="Prediction"
    )

    submit_btn = gr.Button("Analyze News")

    submit_btn.click(
        fn=predict_news,
        inputs=news_input,
        outputs=output
    )

    gr.Examples(
        examples=[
            ["NASA announces new moon mission for 2027."],
            ["Scientists confirm aliens built the pyramids."],
            ["Government launches nationwide AI education program."]
        ],
        inputs=news_input
    )

demo.launch()