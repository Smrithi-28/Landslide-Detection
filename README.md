# AI-Based Landslide Detection Using DeepLabV3+

An end-to-end deep learning project for identifying and segmenting landslide-affected regions from multispectral satellite imagery using **DeepLabV3+**.

The system leverages 14-channel satellite data from the **Landslide4Sense dataset** to perform binary semantic segmentation, generating pixel-level predictions that distinguish landslide-affected areas from surrounding terrain.

### Key Features
- **DeepLabV3+ Architecture:** Encoder-decoder network with Atrous Spatial Pyramid Pooling (ASPP) for extracting multi-scale spatial and contextual features.
- **Multispectral Image Processing:** Processes 14 spectral bands to capture information beyond conventional RGB imagery.
- **Pixel-Level Segmentation:** Generates binary masks to localize landslide-affected regions.
- **Prediction Visualization:** Displays original satellite imagery, predicted segmentation masks, landslide overlays, and probability heatmaps.
- **Model Evaluation:** Includes accuracy, IoU, Dice Score, precision, recall, confusion matrices, and training-validation performance curves.
- **Interactive Web Application:** Built with Streamlit for satellite image uploads, real-time inference, result visualization, and downloadable segmentation masks.

### Technology Stack
Python | TensorFlow | Keras | DeepLabV3+ | NumPy | Matplotlib | Streamlit | HDF5

### Objective
To explore the application of deep learning and multispectral remote sensing in automated landslide identification, supporting faster analysis of potentially affected geographical regions.
