# VPD: Visual Perception Task Model with Improved Referring Image Segmentation

This repository contains the code and models for our project, aiming to improve the referring image segmentation task in the Visual Perception (VP) model. We tried enhance the authors' original model by freezing the stable diffusion module and varying the amount of noise added to the input image. We visualized the attention maps of the model and gave insights on how this model might be improved in future research. This README provides instructions on how to set up and run the code.

## Setting Up the Environment

Before running the code, install the requirements:

```bash

%cd VPD2/refer
pip install -r requirements.txt

%cd VPD2/stablediffusion
pip install -r requirements.txt
```

Training the Frozen VPD Model with Added Noise Scale
Run the following command to start training the frozen VPD model with a specific noise scale value. Note that you can modify the noise scale value directly in the model file within the model_refer folder.

```bash

python VPD2/refer/my_train.py --dataset refcoco --split val --epochs 1 --batch-size 4 --workers 4 --img_size 512
```

Running Inference with Specified Checkpoints
After training, perform inference with a specified checkpoint as follows:

```bash
python VPD2/refer/my_test.py --dataset refcoco --split val --epochs 1 --workers 4 --img_size 512

```

![Alt Text](https://github.com/melvinsevi/MVA-Project-Unleashing-Text-to-Image-Diffusion-Models-for-Visual-Perception/blob/main/VPDgit.png?raw=true)

Additional Information
For more information about the implementation or the VPD model itself, consult the paper, "Visual Perception Task," or visit the official project page.

Please note that some paths mentioned above assume Unix-like systems such as Linux or macOS; adjust accordingly for other operating systems. If you encounter any issues while setting up or running the code, feel free to open an issue in the GitHub repository. Happy coding!

