# Referring Image Segmentation via Text-to-Image Diffusion Models

Welcome to our project repository! Here, we present our efforts to enhance the referring image segmentation task within the VPD model. Our approach involves refining the original model by incorporating a frozen diffusion module while experimenting with different levels of noise added to the input image. We delve into visualizing the attention maps generated by the model, offering valuable insights for potential improvements in future research endeavors. This README provides comprehensive instructions on setting up and running the codebase.

## Setting Up the Environment

To get started, ensure you have the necessary dependencies installed. Navigate to the respective directories and execute the following commands:

```bash

%cd VPD2/refer
pip install -r requirements.txt

%cd VPD2/stablediffusion
pip install -r requirements.txt
```

# Training the Frozen VPD Model with Added Noise Scale

Begin training the frozen VPD model with a specified noise scale value using the following command. You can adjust the noise scale directly within the model file located in the model_refer folder according to your preferences:

```bash

python VPD2/refer/my_train.py --dataset refcoco --split val --epochs 1 --batch-size 4 --workers 4 --img_size 512
```

# Running Inference with Specified Checkpoints

Once training is complete, proceed with performing inference using a specified checkpoint:

```bash
python VPD2/refer/my_test.py --dataset refcoco --split val --epochs 1 --workers 4 --img_size 512

```

# Visualizing Attention Maps

We provide attention maps obtained during inference, showcasing the impact of varying levels of noise added to the input image on model performance:

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/melvinsevi/MVA-Project-Unleashing-Text-to-Image-Diffusion-Models-for-Visual-Perception/blob/main/VPDgit.png?raw=true" alt="Attention Map 1" width="450"/>
    <img src="https://github.com/melvinsevi/MVA-Project-Unleashing-Text-to-Image-Diffusion-Models-for-Visual-Perception/blob/main/VPD2/VPD_img.png?raw=true" alt="Attention Map 2" width="450"/>
</div>
