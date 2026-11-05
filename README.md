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


## Model Performance

### Training Curves
![Training Accuracy](<img width="648" height="379" alt="Screenshot 2026-11-06 014511" src="https://github.com/user-attachments/assets/1f0384b0-676d-4644-9086-729347fc844d" />
)

### Evaluation Results
![Confusion Matrix](<img width="327" height="269" alt="Screenshot 2026-11-06 014458" src="https://github.com/user-attachments/assets/4208ea92-c624-468f-8d32-52a939ce8148" />
)

## Landslide Detection Results

### Prediction Samples
| Original Image | Ground Truth | Predicted Mask |
|---|---|---|
| ![]<img width="321" height="349" alt="Screenshot 2026-11-06 014537" src="https://github.com/user-attachments/assets/b31da028-78e3-4e8a-a452-ca708dbb0ab1" />
) | ![](<img width="318" height="334" alt="Screenshot 2026-11-06 014550" src="https://github.com/user-attachments/assets/fff06f31-04b4-47fc-bb9f-bf2a802146b0" />
) | ![](<img width="638" height="734" alt="Screenshot (45)" src="https://github.com/user-attachments/assets/72f79df1-91c5-4fc9-86be-4cb12faf418a" />
) |

