
# AI-Based Landslide Detection Using DeepLabV3+

An end-to-end deep learning project for identifying and segmenting landslide-affected regions from multispectral satellite imagery using **DeepLabV3+**.

The system leverages 14-channel satellite data from the **Landslide4Sense dataset** to perform binary semantic segmentation, generating pixel-level predictions that distinguish landslide-affected areas from surrounding terrain.

## Key Features

- **DeepLabV3+ Architecture:** Encoder-decoder network with Atrous Spatial Pyramid Pooling (ASPP) for extracting multi-scale spatial and contextual features.
- **Multispectral Image Processing:** Processes 14 spectral bands to capture information beyond conventional RGB imagery.
- **Pixel-Level Segmentation:** Generates binary masks to localize landslide-affected regions.
- **Prediction Visualization:** Displays original satellite imagery, predicted segmentation masks, landslide overlays, and probability heatmaps.
- **Model Evaluation:** Includes accuracy, IoU, Dice Score, precision, recall, confusion matrices, and training-validation performance curves.
- **Interactive Web Application:** Built with Streamlit for satellite image uploads, model inference, result visualization, and downloadable segmentation masks.

## Technology Stack

Python | TensorFlow | Keras | DeepLabV3+ | NumPy | Matplotlib | Streamlit | HDF5

## Objective

To explore the application of deep learning and multispectral remote sensing in automated landslide identification, supporting faster analysis of potentially affected geographical regions.

---

## Model Performance

### Training Curves

<p align="center">
  <img src="https://github.com/user-attachments/assets/1f0384b0-676d-4644-9086-729347fc844d" width="700" alt="Model Training Accuracy Curves"/>
</p>

### Evaluation Results

<p align="center">
  <img src="https://github.com/user-attachments/assets/4208ea92-c624-468f-8d32-52a939ce8148" width="450" alt="Model Confusion Matrix"/>
</p>

---

## Landslide Detection Results

### Prediction Samples

The following examples compare the original satellite imagery, ground-truth segmentation masks, and model-generated predictions.

<table>
  <tr>
    <th width="33%" align="center">Original Satellite Image</th>
    <th width="33%" align="center">Ground Truth Mask</th>
    <th width="33%" align="center">Predicted Segmentation</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b31da028-78e3-4e8a-a452-ca708dbb0ab1" width="100%" alt="Original Satellite Image"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/fff06f31-04b4-47fc-bb9f-bf2a802146b0" width="100%" alt="Ground Truth Landslide Mask"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/72f79df1-91c5-4fc9-86be-4cb12faf418a" width="100%" alt="Predicted Landslide Segmentation"/>
    </td>
  </tr>
</table>

---

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Accuracy:** Measures the overall proportion of correctly classified pixels.
- **Intersection over Union (IoU):** Measures the overlap between predicted and ground-truth segmentation masks.
- **Dice Score:** Evaluates the similarity between predicted and actual landslide regions.
- **Precision:** Measures the proportion of predicted landslide pixels that are correctly identified.
- **Recall:** Measures the proportion of actual landslide pixels detected by the model.

## Application

The Streamlit application provides an interactive interface for uploading multispectral satellite data and visualizing landslide segmentation predictions.

Key outputs include:
- Predicted binary segmentation masks.
- Landslide region overlays.
- Pixel-wise probability heatmaps.
- Downloadable prediction masks.

---

## Dataset

**Landslide4Sense Dataset**

A multispectral remote sensing dataset containing satellite imagery and corresponding landslide segmentation masks.


