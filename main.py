import gradio as gr
from ultralytics import YOLO
import os
import uuid

# Load YOLOv8 model
model = YOLO("best.pt")

# Prediction function
def detect_defects(image):
    input_path = f"temp_{uuid.uuid4()}.jpg"
    image.save(input_path)

    results = model(input_path)
    output_path = f"result_{uuid.uuid4()}.jpg"
    results[0].save(filename=output_path)

    return output_path

# Gradio Interface
demo = gr.Interface(
    fn=detect_defects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="filepath"),
    title="Tyre Defect Detection",
    description="Upload an image of a tyre to detect defects using a YOLOv8 model."
)

if __name__ == "__main__":
    # Get port from environment for Render

    demo.launch(server_name="127.0.0.1", server_port=7860)
