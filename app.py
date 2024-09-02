import streamlit as st
import cv2
from skimage import metrics
from PIL import Image
import numpy as np

def main():
    st.title("Image Comparison using SSIM")

    # Upload images
    uploaded_image1 = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"])
    uploaded_image2 = st.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"])

    if uploaded_image1 and uploaded_image2:
        # Open images
        image1 = Image.open(uploaded_image1)
        image2 = Image.open(uploaded_image2)

        # Convert images to NumPy arrays
        image1_np = np.array(image1)
        image2_np = np.array(image2)

        # Convert images to grayscale
        if len(image1_np.shape) == 3:  # Check if image has color channels
            image1_np = cv2.cvtColor(image1_np, cv2.COLOR_RGB2BGR)
        if len(image2_np.shape) == 3:  # Check if image has color channels
            image2_np = cv2.cvtColor(image2_np, cv2.COLOR_RGB2BGR)

        image1_gray = cv2.cvtColor(image1_np, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2_np, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        ssim_score, _ = metrics.structural_similarity(image1_gray, image2_gray, full=True)

        # Display results
        st.image(image1, caption='First Image', use_column_width=True)
        st.image(image2, caption='Second Image', use_column_width=True)
        st.write(f"SSIM Score: {round(ssim_score, 2)}")

if __name__ == "__main__":
    main()
