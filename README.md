# AI Project - Image colorization using GAN-Unet

I. Introduction
This is a project to build an automatic image colorization model from black and white images using a combination of GAN (Generative Adversarial Networks) and UNET neural network model. With a grayscale image as input, the model produces a colored image as output.Explore the documentation on GAN and Unet to gain a better understanding of these two models and how they synergize with each other.

At this point, we only use the final file of the project with a size (SIZE) of 128x128. However, before that, we trained on this dataset at a size of 64x64. The reason we use two models with different sizes in this project is because the resources of Google Colab are limited and cannot be used to train on large datasets or input sizes that are too large

Fundamentally, the training process for the two models and the image colorization process are identical. We used the 64x64 model as a pre-trained model to optimize resources and enhance output quality for the main model, which is the 128x128 model.

*Required paths:
- Pre-trained 64x64 model: https://drive.google.com/file/d/1Q9TrsqTo3tVY3zACUVW7BLIJIFzE56s2/view?usp=drive_link
- Final (best) 128x128 model: https://drive.google.com/file/d/1hN7mM0onMkbXqCUFp2fGZg2gCy_NBgPi/view?usp=drive_link
- Dataset: https://drive.google.com/drive/folders/1R1cceF-7gmaeG20WTvy2ima-oCptnNbD?usp=drive_link

*Regarding the important files in the project:
static: Location for storing input and output images on the website.
template: Builds the basic interface.
app.py: Uses the Flask library to embed the model into the web


II. Step to Clone the GitHub repository and run the project
  1. Clone the repository
  - git clone https://github.com/ThanhSan97/Image-colorization-using-GAN_UNET.git
  2. Go to folder and Install the necessary libraries with pip
  - pip install -r requirements.txt
  4. Download the 'model.h5' file and save it to the project directory.
  5. Run app.py
  6. Enjoy the game 
