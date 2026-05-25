import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_news(text):
    if not text.strip():
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
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, prediction = torch.max(probs, dim=1)

    prediction = prediction.item()
    confidence = confidence.item() * 100

    if prediction == 1:
        label = "🟢 REAL NEWS"
    else:
        label = "🔴 FAKE NEWS"

    return f"{label}\n\nConfidence Score: {confidence:.2f}%"

css = """
body, .gradio-container {
    background: #0f172a !important;
    color: #f8fafc !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}

h1, h2, h3, p, label, .markdown, .prose {
    color: #f8fafc !important;
}

textarea, input {
    background: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
}

button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
}

footer {
    display: none !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📰 AI Fake News Detector")
    gr.Markdown("Detect whether a news article is REAL or FAKE using a fine-tuned DistilBERT model.")

    news_input = gr.Textbox(
        lines=12,
        placeholder="Paste your news article here...",
        label="News Article"
    )

    output = gr.Textbox(label="Prediction")

    submit_btn = gr.Button("Analyze News")
    submit_btn.click(predict_news, news_input, output)

    gr.Examples(
        examples=[
            ["NASA announces new moon mission for 2027."],
            ["Scientists confirm aliens built the pyramids."],
            ["Government launches nationwide AI education program."]
        ],
        inputs=news_input
    )

demo.launch()