import gradio as gr
import torch

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

model_path = "model"

tokenizer = DistilBertTokenizer.from_pretrained(model_path)

model = DistilBertForSequenceClassification.from_pretrained(model_path)

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

    return f"""
{label}

Confidence Score: {confidence:.2f}%
"""


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 📰 AI Fake News Detector
        
        Detect whether a news article is REAL or FAKE using a fine-tuned DistilBERT model.
        """
    )

    with gr.Row():

        input_text = gr.Textbox(
            lines=15,
            placeholder="Paste news article here...",
            label="News Article"
        )

        output_text = gr.Textbox(
            label="Prediction",
            lines=6
        )

    submit_btn = gr.Button("Analyze News")

    submit_btn.click(
        fn=predict_news,
        inputs=input_text,
        outputs=output_text
    )

demo.launch()