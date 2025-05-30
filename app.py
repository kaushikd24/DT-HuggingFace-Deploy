import gradio as gr
from src.predict import predict_action

def interface_fn(date_input):
    result = predict_action(date_input)
    return result

demo = gr.Interface(
    fn=interface_fn,
    inputs=gr.Textbox(label="Enter Date (YYYY-MM-DD)"),
    outputs="text",
    title="Decision Transformer Market Timer",
    description="Get Buy / Hold / Sell predictions based on market conditions"
)

if __name__ == "__main__":
    demo.launch()
