import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="hamzab/roberta-fake-news-classification"
)

def predict_news(text):

    if not text.strip():
        return "Please enter some news text."

    result = classifier(text)[0]

    label = result["label"]
    confidence = result["score"] * 100

    if label == "LABEL_1":
        prediction = "🟢 REAL NEWS"
    else:
        prediction = "🔴 FAKE NEWS"

    return f"""
{prediction}

Confidence Score: {confidence:.2f}%
"""


custom_css = """
body {
    background-color: #0f172a;
}

.gradio-container {
    max-width: 1000px !important;
    margin: auto;
}

h1 {
    text-align: center;
    color: white;
}

footer {
    visibility: hidden;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    gr.Markdown(
        """
        # 📰 AI Fake News Detector
        
        Detect whether a news article is REAL or FAKE using Hugging Face Transformers.
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

    gr.Examples(
        examples=[
            ["NASA announces new moon mission for 2027."],
            ["Scientists confirm aliens built the pyramids."],
            ["Government launches nationwide AI education program."]
        ],
        inputs=input_text
    )

demo.launch()