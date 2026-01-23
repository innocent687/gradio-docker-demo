import gradio as gr
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(sepal_length, sepal_width, petal_length, petal_width):
    x = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(x)[0]
    return f"Predicted class: {pred}"

inputs = [
    gr.Number(label="Sepal Length"),
    gr.Number(label="Sepal Width"),
    gr.Number(label="Petal Length"),
    gr.Number(label="Petal Width"),
]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Iris Classifier Demo",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )