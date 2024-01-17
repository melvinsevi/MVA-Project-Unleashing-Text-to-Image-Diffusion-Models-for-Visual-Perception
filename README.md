VPD: Visual Perception Task Model with Improved Referring Image Segmentation
This repository contains the code and models for our project, which aims to improve the referring image segmentation task in the Visual Perception (VP) model. We build upon the authors' original model by freezing the stable diffusion module and varying the amount of noise added to the input image. This README provides instructions on how to set up and run the code.

Setting Up the Environment
Before running the code, create a virtual environment and activate it:

python3 -m venv env
source env/bin/activate   # On Windows, use ".\env\Scripts\activate" instead
Install the required dependencies for both refer and stablediffusion modules:

%cd VPD2/refer
!pip install -r requirements.txt

%cd VPD2/stablediffusion
!pip install -r requirements.txt
Training the Frozen VPD Model with Added Noise Scale
Run the following command to start training the frozen VPD model with a specific noise scale value. Note that you can modify the noise scale value directly in the model file within the model_refer folder.

!python VPD2/refer/my_train.py --dataset refcoco --split val --epochs 1 --batch-size 4 --workers 4 --img_size 512
Running Inference with Specified Checkpoints
After training, perform inference with a specified checkpoint as follows:

!python VPD2/refer/my_test.py --dataset refcoco --split val --epochs 1 --workers 4 --img_size 512
Structure of the Repository
Here's an overview of the structure of the repository:

VPD2/: Contains the source code for the VPD model and related components.
refer/: Directory containing scripts and code for the referring image segmentation part of the VPD model.
data/: Dataset loading utilities and preprocessing functions.
models/: Implementations of various VPD architectures.
utils/: Miscellaneous utility functions.
my_train.py: Main script to train the referred VPD model.
my_test.py: Main script to test the referred VPD model with saved checkpoints.
stablediffusion/: Directory containing scripts and code for the stable diffusion component of the VPD model.
configs/: Configuration files for different experimental settings.
results/: Output logs, trained models, and evaluation metrics from the experiments.
Additional Information
For more information about the implementation or the VPD model itself, consult the paper, "Visual Perception Task" or visit the official project page.

Please note that some paths mentioned above assume Unix-like systems such as Linux or macOS; adjust accordingly for other operating systems. If you encounter any issues while setting up or running the code, feel free to open an issue in the GitHub repository. Happy coding!
