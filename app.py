import gradio as gr
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Map class numbers to names
class_names = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Define prediction function
def predict(sepal_length, sepal_width, petal_length, petal_width):
    x = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    class_index = model.predict(x)[0]
    class_name = class_names.get(class_index, "Unknown class")
    return f"Predicted class: {class_name}"

# Gradio interface
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
    demo.launch(server_name="0.0.0.0", server_port=7860)