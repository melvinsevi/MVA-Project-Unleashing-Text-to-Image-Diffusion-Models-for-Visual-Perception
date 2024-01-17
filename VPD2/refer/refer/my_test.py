import torch
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt

from models_refer.model import VPDRefer  # Import your model class
from transformers.models.clip.modeling_clip import CLIPTextModel

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((your_image_size, your_image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, image_tensor

def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def test_single_image(model, clip_model, image_path, device):
    model.eval()

    # Load and show the original image
    original_image, image_tensor = load_image(image_path)
    show_image(original_image)

    # Modify the following lines based on your model input requirements
    # For example, you may need to modify the text input depending on your model's expectations.
    sentences = torch.tensor([your_text_embedding]).unsqueeze(0).to(device)
    attentions = torch.tensor([your_attention_map]).unsqueeze(0).to(device)

    embedding = clip_model(input_ids=sentences).last_hidden_state
    output = model(image_tensor.to(device), embedding)

    # Process the output as needed for your specific case

if __name__ == "__main__":
    # Load your model and CLIP model
    single_model = VPDRefer(sd_path='../checkpoints/v1-5-pruned-emaonly.ckpt', neck_dim=[320, 640 + args.token_length, 1280 + args.token_length, 1280])
    checkpoint = torch.load('path/to/your/checkpoint.pth', map_location='cpu')  # Provide the path to your checkpoint file
    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)

    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.cuda()
    clip_model = clip_model.eval()

    # Test on a single image
    image_path = "path/to/your/image.jpg"
    test_single_image(model, clip_model, image_path, device)
