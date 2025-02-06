# AI-Powered Image-to-Image Transformation with Stable Diffusion

Welcome to the **AI-Powered Image-to-Image Transformation** project, where you can use a pre-trained *Stable Diffusion* model to transform your images based on descriptive text prompts. This project leverages state-of-the-art deep learning techniques to modify an input image according to a user-defined concept, blending creativity with AI.

**This notebook is part of an older project from 2023, where we leveraged the Stable Diffusion model to perform Image-to-Image (Img2Img) transformations. At the time, this approach allowed us to take any input image and creatively manipulate it based on a text description. Using cutting-edge deep learning models, we were able to experiment with how AI could reimagine images with just a few lines of code. Though the project is from 2023, the techniques explored here are still quite relevant for creative image manipulation today.**

In this README, we will go through each part of the project, from setting up the environment to the code and an explanation of how everything works under the hood.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [How to Use](#how-to-use)
4. [Code Walkthrough](#code-walkthrough)
   - [1. Installing Dependencies](#1-installing-dependencies)
   - [2. Loading the Stable Diffusion Model](#2-loading-the-stable-diffusion-model)
   - [3. Image Upload and Processing](#3-image-upload-and-processing)
   - [4. Image Generation](#4-image-generation)
5. [Understanding the Model](#understanding-the-model)
6. [Example Use Cases](#example-use-cases)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

This project demonstrates how to use *Stable Diffusion*, a text-to-image model, for **Image-to-Image (Img2Img)** transformations. By providing an input image and a descriptive text prompt, the model generates a modified image that aligns with the prompt while maintaining the core structure of the original image.

For example, you can upload a picture of a landscape and use a prompt like "a magical sunset over a mountain range," and the model will generate a transformed image based on the description, while keeping the landscape intact.

---

## Installation

1. **Install the required Python libraries**:

    ```
    pip install transformers diffusers torch accelerate matplotlib huggingface_hub
    ```

2. **If using Google Colab**, simply open the notebook and run each cell sequentially.

Once the installation is complete, you're ready to begin transforming images using the power of AI!

---

## How to Use

1. **Upload an Image**: Use the upload prompt to bring in an image from your local machine or Google Drive.

2. **Define a Text Prompt**: Provide a text prompt that describes the changes you want to see in the image. The more specific the prompt, the more creative the result.

3. **Generate the Image**: Run the function to generate the transformed image, which is then displayed and saved for your use.

4. **Download the Result**: Once the image is generated, you can save and download it for further use or sharing.

---

## Code Walkthrough

Now, let’s break down each part of the code to explain its functionality and how everything works together to bring the image transformation process to life.

### 1. Installing Dependencies

Before we get into the actual code, we need to install several key libraries that enable the image generation process:

- **transformers**: This library is used to load the pre-trained _Stable Diffusion_ model and handle various pipelines for text and image processing.
- **diffusers**: The core library for accessing the _Stable Diffusion_ model and generating images.
- **torch**: The deep learning framework that powers the model, providing tensor operations and GPU support.
- **accelerate**: Ensures the model runs efficiently on GPUs, significantly speeding up the process.
- **matplotlib**: A library for displaying the generated images.
- **huggingface_hub**: This helps download pre-trained models directly from Hugging Face’s repository.

### 2. Loading the Stable Diffusion Model

The core of this notebook is the _Stable Diffusion_ model. We use the `StableDiffusionPipeline` class from the **diffusers** library to load the pre-trained model. This model is designed for text-to-image generation, and in our case, it’s also capable of transforming an image based on an additional prompt.

```
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", 
                                               torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Move model to GPU for faster processing
```

```StableDiffusionPipeline.from_pretrained()``` loads the model weights, enabling us to use the pre-trained network to generate images. The ```"stabilityai/stable-diffusion-2-1-base"``` specifies the model variant, which is optimized for general image generation.
```.to("cuda")``` moves the model to the GPU, which drastically accelerates the image generation process.
### 3. Image Upload and Processing
To enable users to upload their own input images, we define a function ```upload_image()``` that opens a file upload dialog (useful for Google Colab) and loads the image:

```
def upload_image():
    uploaded = files.upload()
    for image_name in uploaded.keys():
        img = Image.open(image_name)
    return img
```

The uploaded image is returned as a PIL image, ready for resizing and processing by the model.

### 4. Image Generation
Once the image is uploaded, we apply a transformation using the ```generate_image_from_input_image()``` function. This function first resizes the image to fit the model’s expected input size ```(512x512 pixels)``` and then passes it through the model’s pipeline.

```
def generate_image_from_input_image(prompt, input_image):
    input_image = input_image.resize((512, 512))  # Resize to model's input size
    image = pipe(prompt=prompt, init_image=input_image, strength=0.75).images[0]
    return image
```

```strength=0.75``` controls how much the model should rely on the original image versus the prompt. A lower strength results in less modification, while a higher strength leads to a more creative transformation.
```pipe()``` generates the transformed image, blending the input image with the text prompt to create the new version.

 Understanding the Model
The Stable Diffusion model is a latent diffusion model that works by operating on a compressed (latent) representation of the image. This makes it computationally efficient, allowing high-quality image generation with fewer resources compared to working with high-resolution pixel-based images directly.

- **Latent Space:** Instead of manipulating high-resolution images, the model works on a lower-dimensional latent space, generating a representation that is then decoded back into a high-resolution image.

- **Text Encoder:** A CLIP model is used to encode the text prompt, which is then used to guide the image generation process.
The model learns to iteratively refine images, starting from random noise and gradually transforming them into something meaningful based on the given prompt and initial image.

# Example Use Cases
**Image Restoration:** Start with an old, blurry image and provide a prompt like “clear and sharp photo” to restore the image quality.

**Style Transfer:** Upload a photo and use prompts like “turn this into a painting in the style of Van Gogh.”

**Fantasy or Sci-Fi Imagery** Transform everyday images into fantastical landscapes or sci-fi scenes by using creative prompts.

# Contributing
We welcome contributions to improve this project! If you find any issues or have ideas for enhancements, feel free to open an issue or submit a pull request.

### Steps to Contribute:
1. **Fork this repository.**

2. **Create a new branch (git checkout -b feature-name).**

3. **Make your changes.**

4. **Commit your changes (git commit -am 'Add new feature').**

5. **Push to the branch (git push origin feature-name).**

6. **Open a pull request.**

# License
This project is licensed under the MIT License - see the LICENSE file for details.

**That's it! You now have everything you need to start generating images using Stable Diffusion in this project. Experiment with your own images and text prompts, and let the power of AI creativity take your visuals to new heights!**
