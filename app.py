import gradio as gr
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# -----------------------------
# Load Model + Tokenizer
# -----------------------------

model_path = "model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)

model.eval()

# -----------------------------
# Prediction Function
# -----------------------------

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

        probabilities = torch.nn.functional.softmax(
            outputs.logits,
            dim=-1
        )

    # Label Mapping
    # 0 = Real
    # 1 = Fake

    real_score = probabilities[0][0].item()
    fake_score = probabilities[0][1].item()

    # Better Threshold Logic

    if fake_score > 0.80:

        label = "🔴 LIKELY FAKE NEWS"
        confidence = fake_score * 100

    elif real_score > 0.80:

        label = "🟢 LIKELY REAL NEWS"
        confidence = real_score * 100

    else:

        label = "🟡 UNCERTAIN / MIXED SIGNALS"
        confidence = max(real_score, fake_score) * 100

    return f"""
{label}

Confidence Score: {confidence:.2f}%

Fake Score: {fake_score:.2f}
Real Score: {real_score:.2f}

Note:
This model predicts whether text resembles patterns commonly found in fake or real news articles.
"""

# -----------------------------
# Custom UI Styling
# -----------------------------

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

# -----------------------------
# Gradio UI
# -----------------------------

with gr.Blocks(
    css=css,
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
# 📰 AI Fake News Detector

This AI model analyzes whether a news article resembles patterns commonly found in fake or real news articles.
"""
    )

    news_input = gr.Textbox(
        lines=12,
        placeholder="Paste a news article here...",
        label="News Article"
    )

    output = gr.Textbox(
        label="Prediction"
    )

    submit_btn = gr.Button(
        "Analyze News"
    )

    submit_btn.click(
        predict_news,
        news_input,
        output
    )


gr.Examples(
    examples=[
        [
            "The Reserve Bank of India kept interest rates unchanged during its latest monetary policy meeting. Officials cited stable inflation and steady economic growth as key reasons behind the decision. Analysts expect the central bank to continue monitoring global financial conditions closely."
        ],
        [
            "Scientists confirmed that humans can survive underwater for several hours after drinking a newly discovered herbal liquid. Researchers claim the formula changes lung function permanently and allows direct oxygen absorption from water."
        ]
    ],
    inputs=news_input
)



demo.launch()