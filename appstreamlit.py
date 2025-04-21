import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import cv2
import glob  
import os
import gc
import torch.nn as nn
import torchvision.models as models
from cnn_builder import UNET
from torchvision.models import efficientnet_b1
from auxiliar_functions import unet_preprocess, unet_inference, infer_single_image, apply_binary_mask
# --- Helper Functions (Adapt these to your specific models) ---
import platform
import psutil

def bytes_to_mb(bytes):
    """Converts bytes to megabytes."""
    return bytes / (1024 * 1024)
    
def concatenate_file_chunks(chunk_prefix, output_filepath):
    """Concatenates the split file chunks."""
    chunk_files = sorted(glob.glob(f"{chunk_prefix}*"))
    if not chunk_files:
        #st.error(f"Error: No chunk files found with prefix '{chunk_prefix}'.")
        return False
    try:
        with open(output_filepath, 'wb') as outfile:
            for chunk_file in chunk_files:
                with open(chunk_file, 'rb') as infile:
                    outfile.write(infile.read())
        #st.info(f"Successfully concatenated the chunks into '{output_filepath}'.")
        return True
    except Exception as e:
        #st.error(f"An error occurred during concatenation: {e}")
        return False

##################################################################################
########################## Unet Model Initialization #############################
##################################################################################

@st.cache_resource
def load_unet():
    unet_filename = "unet_gc_dice_0-9020.pth.tar"
    unet_pth = os.path.join(unet_filename)
    unet_chunk_prefix = os.path.join("unet_gc_dice_0-9020_part_")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet_loaded_model = None  # Initialize to None

    # Check if the full UNET model file exists
    if not os.path.exists(unet_pth):
        #st.info("UNET model file not found. Attempting to concatenate chunks...")
        if concatenate_file_chunks(unet_chunk_prefix, unet_pth):
            unet_loaded_model = UNET(in_channels=3, out_channels=1).to(device)
            try:
                checkpoint = torch.load(unet_pth, map_location=torch.device(device))
                unet_loaded_model.load_state_dict(checkpoint["state_dict"])
                unet_loaded_model.eval()
                #st.success("UNET model loaded successfully after concatenation.")
            except Exception as e:
                #st.error(f"Error loading concatenated UNET model: {e}")
                st.stop()
        else:
            #st.error("Failed to concatenate UNET model chunks. Please ensure all parts are present.")
            st.stop()
    else:
        # Load the UNET model directly if it exists
        unet_loaded_model = UNET(in_channels=3, out_channels=1).to(device)
        try:
            checkpoint = torch.load(unet_pth, map_location=torch.device(device))
            unet_loaded_model.load_state_dict(checkpoint["state_dict"])
            unet_loaded_model.eval()
            #st.info("UNET model loaded successfully.")
        except Exception as e:
            #st.error(f"Error loading UNET model: {e}")
            st.stop()

    return unet_loaded_model

unet_loaded_model = load_unet()

@st.cache_resource
def load_efficientnet():
    effnet_filename = "efficientnet_b1_pretained_experiment5.pth.tar"
    effnet_pth = os.path.join(effnet_filename)
    effnet_chunk_prefix = os.path.join("efficientnet_b1_pretained_experiment5_part_")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    effnet_loaded_model = None  # Initialize to None

    # Check if the full EffNet model file exists
    if not os.path.exists(effnet_pth):
        #st.info("EffNet model file not found. Attempting to concatenate chunks...")
        if concatenate_file_chunks(effnet_chunk_prefix, effnet_pth):
            effmodel = models.efficientnet_b1(weights=None) # Load with no pretrained weights initially
            num_features = effmodel.classifier[-1].in_features
            effmodel.classifier[-1] = nn.Linear(num_features, 3)
            try:
                checkpoint = torch.load(effnet_pth, map_location=torch.device(device))
                effnet_loaded_model = effmodel # Assign the model instance
                effnet_loaded_model.load_state_dict(checkpoint)
                effnet_loaded_model.eval()
                #st.success("EffNet model loaded successfully after concatenation.")
            except Exception as e:
                #st.error(f"Error loading concatenated EffNet model: {e}")
                st.stop()
        else:
            #st.error("Failed to concatenate EffNet model chunks. Please ensure all parts are present.")
            st.stop()
    else:
        # Load the EffNet model directly if it exists
        effmodel = models.efficientnet_b1(weights=None) # Load with no pretrained weights initially
        num_features = effmodel.classifier[-1].in_features
        effmodel.classifier[-1] = nn.Linear(num_features, 3)
        try:
            checkpoint = torch.load(effnet_pth, map_location=torch.device(device))
            effnet_loaded_model = effmodel # Assign the model instance
            effnet_loaded_model.load_state_dict(checkpoint)
            effnet_loaded_model.eval()
            #st.info("EFFNET model loaded successfully.")
        except Exception as e:
            #st.error(f"Error loading EffNET model: {e}")
            st.stop()

    return effnet_loaded_model

#effnet_loaded_model = load_efficientnet()

##################################################################################
########################## Classifier Model Initialization #######################
##################################################################################
@st.cache_resource
def load_shufflenet():
    classifier_pth = "shufflenet_v2_x2_0_lr0-001_epoch30_pretrained.pth.tar"
    shufflenet_loaded_model = models.shufflenet_v2_x2_0()
    num_features = shufflenet_loaded_model.fc.in_features
    shufflenet_loaded_model.fc = nn.Linear(num_features, 3)  # Replace the final layer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(classifier_pth):
        #st.error(f"Error: ShuffleNet checkpoint file not found at {classifier_pth}")
        return None

    try:
        checkpoint2 = torch.load(classifier_pth, map_location=torch.device(device))
        shufflenet_loaded_model.load_state_dict(checkpoint2)
        shufflenet_loaded_model.eval()
        #st.info("ShuffleNet model loaded successfully.")
        return shufflenet_loaded_model
    except Exception as e:
        st.error(f"Error loading ShuffleNet model: {e}")
        st.error(f"Details: {e}") # Print the specific error message
        return None

#shufflenet_loaded_model = load_classifier()


st.title("Fundus images disease classification")


# Sidebar for model selection
st.sidebar.header("Model Selection")
classifier_model_name = st.sidebar.selectbox(
    "Choose a Classification Model:",
    ["EfficientNet", "ShuffleNet"]
)

# Load the selected classifier model
if classifier_model_name == "EfficientNet":
    effnet_loaded_model = load_efficientnet()
    shufflenet_loaded_model = None # Ensure the other model isn't used
elif classifier_model_name == "ShuffleNet":
    shufflenet_loaded_model = load_shufflenet()
    effnet_loaded_model = None # Ensure the other model isn't used
else:
    effnet_loaded_model = None
    shufflenet_loaded_model = None
    st.warning("No classification model selected.")

# st.sidebar.subheader("Load Example Image")
st.sidebar.header("Load Example Image")
example_images = [None] + [f for f in os.listdir() if f.endswith(('.png'))]
selected_example = st.sidebar.selectbox("Choose an example image:", example_images)

st.sidebar.subheader("Or select None and then upload your own image")

uploaded_file = st.file_uploader("Upload your own image:", type=["jpg", "jpeg", "png"])

if selected_example != "None" and selected_example:
    uploaded_file = None
    try:
        aviso = "Images ending with A belong to macular degeneration class (AMD), ending with D belong to diabetic retinopathy (DR), and ending with N belong to healthy patients (Normal). Final predicted label is shown at the end of the page"
        st.write(f"**Nota:** {aviso}")
        example_image_path = os.path.join(selected_example)
        original_image = Image.open(example_image_path).convert("RGB")
        resized_original = original_image.resize((512, 512))
        st.image(resized_original, caption=f"Example Image: {selected_example}", use_container_width=True)

        # --- Segmentation ---
        st.subheader("Blood Vessel Segmentation Phase")
        resized_img = unet_preprocess(original_image)
        binary_mask = unet_inference(resized_img, unet_loaded_model)

        # Visualization of mask
        st.image(binary_mask, caption="Predicted Blood Vessels Binary Mask", use_container_width=True)

        # Apply mask to the original image
        segmented_image = apply_binary_mask(resized_original, binary_mask)
        st.image(segmented_image, caption="Segmented Image", use_container_width=True)

        # --- Classification ---
    # --- Classification ---
        st.subheader("Classification")
        if classifier_model_name == "EfficientNet" and effnet_loaded_model:
            predicted_class = infer_single_image(effnet_loaded_model, segmented_image)
        elif classifier_model_name == "ShuffleNet" and shufflenet_loaded_model:
            predicted_class = infer_single_image(shufflenet_loaded_model, segmented_image)
            
        if predicted_class == 0:
            predicted_label = "AMD"
        elif predicted_class == 1:
            predicted_label = "DR"
        elif predicted_class == 2:
            predicted_label = "Normal"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"**Predicted Class:** {predicted_label}")
        st.write(f"**This experiment was executed using** {device}")

        
        del resized_original
        del resized_img
        del binary_mask
        del segmented_image
    except Exception as e:
        st.error(f"An error occurred while processing the example image: {e}")

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
        resized_original = original_image.resize((512, 512))
        st.image(resized_original, caption="Uploaded image", use_container_width=True)


        # --- Segmentation ---
        st.subheader("Blood Vessel Segmentation Phase")
        resized_img = unet_preprocess(original_image)
        binary_mask = unet_inference(resized_img, unet_loaded_model)

        # Visualization of mask
        st.image(binary_mask, caption="Predicted Blood Vessels Binary Mask", use_container_width=True)

        # Apply mask to the original image
        segmented_image = apply_binary_mask(resized_original,binary_mask)
        st.image(segmented_image, caption="Segmented Image", use_container_width=True)

        # --- Classification ---
        st.subheader("Classification")
        if classifier_model_name == "EfficientNet" and effnet_loaded_model:
            predicted_class = infer_single_image(effnet_loaded_model, segmented_image)
        elif classifier_model_name == "ShuffleNet" and shufflenet_loaded_model:
            predicted_class = infer_single_image(shufflenet_loaded_model, segmented_image)

        
        if predicted_class == 0:
            predicted_label = "AMD"
        elif predicted_class == 1:
            predicted_label = "DR"
        elif predicted_class == 2:
            predicted_label = "Normal"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"**Predicted Class:** {predicted_label}")
        st.write(f"**This experiment was executed using** {device}")
        


        # Explicitly delete large variables after processing
        del resized_original
        del resized_img
        del binary_mask
        del segmented_image
        del uploaded_file
        gc.collect()

    except Exception as e:
        st.error(f"An error occurred while processing the example image: {e}")
