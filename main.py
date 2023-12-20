import gradio as gr
import torch
import os
from PIL import Image
from torchvision import transforms
import torchvision.models as models

# Load the trained model
model = torch.load('assets/flower_model_vgg19.pth')
model.eval()
# Load labels
with open('assets/flower_dataset_labels.txt', 'r') as f:
    labels=f.readlines()

# Defining the function that's doing the prediction
def predict(inp):
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(102)}
  return confidences

# Preparing texts to be added to the interface
title = "United Kingdom Flower Classification App"
description = (
  "This app allows users to input an image (in jpeg format) of a flower and get a prediction on what type of flower it is from a machine learning model. Top three most likely flower types according to the model's prediction are shown. "
  "The machine learning model was trained to predict flowers occuring in the United Kingdom so it is unlikely to be good at flowers from other parts of the world."
)
ref = """ 
  <div style='line-height: 1;'>
  <hr>
  <h3>App source code</h3>
  <p>The source code of this app <a href='https://github.com/ScilifelabDataCentre/gradio-flower-classification'>can be found on GitHub</a> with an open source license so feel free to use it to build your own apps. The app was built using the <a href='https://www.gradio.app/'>Gradio framework</a>.</p>
  <p>We welcome suggestions and contributions to this app. If you found a mistake or would like to make a suggestion, please create an issue in <a href='https://github.com/ScilifelabDataCentre/gradio-flower-classification'>the app's Github repository</a>. Those who wish are also welcome to submit pull requests.</p>
  <h3>Model behind the app</h3>
  <p>The task of the machine learning problem was to predict the correct flower category from an image. Information about the model:</p>
  <ul>
  <li>We used a CNN model with the VGG19 architecture.</li>
  <li>We used a pre-trained model and transfer learning.</li>
  <li>We used the Adam (or AdamW) optimizer (Adaptive Moment Estimation).</li>
  <li>We split training set into training and validation using a 70-30 random split.</li>
  </ul>
  <p>For more information about the model, including the code used, see <a href='https://github.com/ScilifelabDataCentre/serve-tutorials/tree/main/Webinars/2023-Using-containers-on-Berzelius/flowers-classification'>https://github.com/ScilifelabDataCentre/serve-tutorials/tree/main/Webinars/2023-Using-containers-on-Berzelius/flowers-classification</a>.</p>
  <h3>Training data behind the model</h3>
  <p>Our model was trained on a dataset of images consisting of 102 flower categories. The flowers in the training set are commonly occuring flowers in the United Kingdom. Each class consists of between 40 and 258 images. Total number of images in the dataset used for model training and testing: 2040.</p>
  <p>For information about the dataset used, see <a href='https://www.robots.ox.ac.uk/~vgg/data/flowers/102/'>https://www.robots.ox.ac.uk/~vgg/data/flowers/102/</a>.</p>
  <p>This flower image database was used in:</p>
  <p>Nilsback, M-E. and Zisserman, A. Automated flower classification over a large number of classes.<br>
  <i>Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing</i> (2008)<br>
  http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback08.{pdf,ps.gz}.</p>
  <div>
"""
# Gradio interface definition
interface = gr.Interface(fn=predict,
                         inputs=gr.Image(type="pil"),
                         outputs=gr.Label(num_top_classes=3),
                         allow_flagging="never",
                         title=title,
                         description=description, 
                         article=ref,
                         examples=[
                                os.path.join(os.path.dirname(__file__), "assets/example_images/image_00121.jpg"),
                                os.path.join(os.path.dirname(__file__), "assets/example_images/image_00332.jpg"),
                                os.path.join(os.path.dirname(__file__), "assets/example_images/image_06743.jpg")
                                ]
             )

# Launching Gradio
interface.launch(server_name="0.0.0.0", server_port=8080)
