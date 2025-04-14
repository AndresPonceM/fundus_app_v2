import numpy as np
import cv2
from PIL import Image
import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def unet_preprocess(image):
    img_np = np.array(image)
    green_channel = img_np[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_green = clahe.apply(green_channel)
    equalized = Image.fromarray(equalized_green)
    return equalized.resize((512, 512))

def unet_inference(image, model):
    transform = A.Compose([
        A.Resize(height=512, width=512),  # Pass height and width as separate keyword arguments
        ToTensorV2()])

    img_np = np.array(image.convert("RGB"))
    img = transform(image=np.array(img_np))['image']
    img = img.type(torch.float32).to(device) # Change to .type(torch.float32)
    img = img.unsqueeze(0)

    with torch.no_grad():
        preds = torch.sigmoid(model(img))
        preds = (preds > 0.5).float()
        preds_np = preds.squeeze(0).cpu().numpy()
        preds_np_uint8 = (preds_np * 255).astype(np.uint8).squeeze(0) #added squeeze(0)
        mask_image = Image.fromarray(preds_np_uint8) #create pil image
    return mask_image

def infer_single_image(model, image):

    image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3199, 0.2240, 0.1609], std=[0.3020, 0.2183, 0.1741]),
        ])

    image_transformed = image_transform(image)
    image_transformed_sq = torch.unsqueeze(image_transformed, dim=0)

    with torch.no_grad():
        image_transformed_sq = image_transformed_sq.to(device)
        output = model(image_transformed_sq)
        _, predicted_class_index = torch.max(output.data, 1)


    return predicted_class_index.item() #return the item, not the tensor.

def apply_binary_mask(image, mask):
    mask_np = np.array(mask) / 255.0  # Normalize to 0-1
    inverted_mask_np = 1 - mask_np #Invert the mask
    original_np = np.array(image)
    segmented_np = original_np * np.expand_dims(inverted_mask_np, axis=-1)  # Apply mask
    return Image.fromarray(segmented_np.astype(np.uint8))