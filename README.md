# United Kingdom Flower Classification App

This repository contains code for a web app that allows users to input an image of a flower and get a prediction on what type of flower it is from a machine learning model. The app is built using the [Gradio](https://github.com/gradio-app/gradio) framework. It takes an image as user input (user uploads an image from their computer) and outputs flower types with top three classification scores. The machine learning model is trained to predict flowers occuring in the United Kingdom.

The live app can be found here: [flower-classification.serve.scilifelab.se](https://flower-classification.serve.scilifelab.se/).

## Model behind the app

The task of the machine learning problem was to predict the correct flower category from an image. Information about the model:
- We used a CNN model with the VGG19 architecture.
- We used a pre-trained model and transfer learning.
- We used the Adam (or AdamW) optimizer (Adaptive Moment Estimation).
- We split training set into training and validation using a 70-30 random split.

For more information about the model, including the code used, see [https://github.com/ScilifelabDataCentre/serve-tutorials/tree/main/Webinars/2023-Using-containers-on-Berzelius/flowers-classification](https://github.com/ScilifelabDataCentre/serve-tutorials/tree/main/Webinars/2023-Using-containers-on-Berzelius/flowers-classification).

## Training data

Our model was trained on a dataset of images consisting of 102 flower categories. The flowers in the training set are commonly occuring flowers in the United Kingdom. Each class consists of between 40 and 258 images. Total number of images in the dataset used for model training and testing: 2040.

For information about the dataset used, see [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).

This flower image database was used in:

Nilsback, M-E. and Zisserman, A. Automated flower classification over a large number of classes.
Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (2008) 
http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback08.{pdf,ps.gz}.

## Contributing

We welcome suggestions and contributions. If you found a mistake or would like to make a suggestion, please create an issue in this repository. Those who wish are also welcome to submit pull requests.

## Contact

This dashboard was built by [SciLifeLab Data Centre](https://github.com/ScilifelabDataCentre) team members.
