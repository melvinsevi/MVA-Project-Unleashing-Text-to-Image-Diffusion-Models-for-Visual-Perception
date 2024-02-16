# Referring Image Segmentation via Text-to-Image Diffusion Models

Introduction
------------

In this project, we explore the use of text-to-image diffusion models for improving the referring image segmentation task in the Visual Perception (VP) model. Our approach involves freezing the stable diffusion module and varying the amount of noise added to the input image. We evaluate the performance of our model using the RefCOCO dataset and provide insights into how this model can be further improved in future research.

Getting Started
---------------

To get started, follow these steps:

1. Clone the repository: `git clone https://github.com/melvinsevi/VP-Diffusion-Model.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the RefCOCO dataset: `curl -O http://msvocds.com/refcoco/download.html`

Training the Model
-----------------

To train the frozen VPD model with a specific noise scale value, run the following command:
```bash
python VPD2/refer/my_train.py --dataset refcoco --split val --epochs 1 --batch-size 4 --workers 4 --img_size 512 --noise_scale 0.1
Running Inference with Specified Checkpoints
After training, perform inference with a specified checkpoint as follows:

```bash
python VPD2/refer/my_test.py --dataset refcoco --split val --epochs 1 --workers 4 --img_size 512

```

<div style="text-align:center;">
    <img src="https://github.com/melvinsevi/MVA-Project-Unleashing-Text-to-Image-Diffusion-Models-for-Visual-Perception/blob/main/VPDgit.png?raw=true" alt="Alt Text" width="450"/>
</div>

Here is the attention maps obtained in inference, when varying the noise added to the input image of the model:

<div style="text-align:center;">
    <img src="https://github.com/melvinsevi/MVA-Project-Unleashing-Text-to-Image-Diffusion-Models-for-Visual-Perception/blob/main/VPD2/VPD_img.png?raw=true" alt="Alt Text" width="450"/>
</div>
