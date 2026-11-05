import streamlit as st
import numpy as np
import tensorflow as tf
import h5py
import json
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Landslide Detection AI",
    page_icon="🌍",
    layout="wide"
)


# =========================================================
# THEME-AWARE CUSTOM CSS
# Uses Streamlit's active theme variables, so the app adapts
# automatically to both Light and Dark mode.
# =========================================================

st.markdown(
    """
    <style>
        /* Main application surfaces */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        [data-testid="stSidebar"] {
            background-color: var(--secondary-background-color);
        }

        /* Keep normal text readable in either theme */
        .stMarkdown,
        .stMarkdown p,
        .stMarkdown li,
        label,
        [data-testid="stCaptionContainer"] {
            color: var(--text-color);
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text-color);
        }

        /* Hero banner */
        .hero {
            padding: 30px;
            border-radius: 15px;
            background: linear-gradient(120deg, #123c69, #1d7874);
            color: #ffffff !important;
            margin-bottom: 25px;
        }

        .hero h1 {
            color: #ffffff !important;
            font-size: 36px;
        }

        .hero p {
            color: #e8f1f5 !important;
            font-size: 16px;
        }

        /* Metric cards follow the selected theme */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid var(--secondary-background-color);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: var(--text-color);
        }

        /* File uploader and other widgets */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] section {
            color: var(--text-color);
        }

        /* Download button */
        .stDownloadButton button {
            border-radius: 8px;
        }

        /* Improve horizontal separators in both themes */
        hr {
            border-color: var(--text-color);
            opacity: 0.2;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FILE PATHS
# Keep these paths aligned with your project folder structure.
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "deeplabv3plus_landslide.keras"
STATS_PATH = BASE_DIR / "data" / "normalization_stats.npz"
HISTORY_PATH = BASE_DIR / "data" / "training_history.json"
EVALUATION_PATH = BASE_DIR / "data" / "evaluation_data.npz"


# =========================================================
# THEME SETTINGS FOR MATPLOTLIB
# Streamlit's theme and Matplotlib charts are styled separately.
# =========================================================

def get_plot_theme():
    """Return readable Matplotlib colors for the active Streamlit theme."""
    try:
        active_theme = st.get_option("theme.base")
    except Exception:
        active_theme = "light"

    if active_theme == "dark":
        return {
            "figure": "#0e1117",
            "axes": "#0e1117",
            "text": "#f0f2f6",
            "grid": "#555b66",
            "spine": "#8a9099"
        }

    return {
        "figure": "#ffffff",
        "axes": "#ffffff",
        "text": "#262730",
        "grid": "#d5d8de",
        "spine": "#777777"
    }


def style_axes(ax):
    """Apply theme-aware colors to a Matplotlib axes."""
    colors = get_plot_theme()
    ax.set_facecolor(colors["axes"])
    ax.tick_params(colors=colors["text"])
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.title.set_color(colors["text"])

    for spine in ax.spines.values():
        spine.set_color(colors["spine"])

    ax.grid(color=colors["grid"], alpha=0.3)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# =========================================================
# LOAD NORMALIZATION STATISTICS
# =========================================================

@st.cache_data
def load_normalization():
    stats = np.load(STATS_PATH)

    band_mean = stats["band_mean"]
    band_std = stats["band_std"]

    return band_mean, band_std


# =========================================================
# LOAD TRAINING HISTORY
# =========================================================

@st.cache_data
def load_history():
    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        history = json.load(file)

    return history


# =========================================================
# LOAD CONFUSION MATRIX
# =========================================================

@st.cache_data
def load_evaluation():
    evaluation = np.load(EVALUATION_PATH)

    cm_raw = evaluation["cm_raw"]
    cm_normalized = evaluation["cm_normalized"]

    return cm_raw, cm_normalized


# =========================================================
# FIND IMAGE DATASET INSIDE HDF5 FILE
# =========================================================

def find_image_dataset(h5_file):
    if "img" in h5_file:
        return h5_file["img"][:]

    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    h5_file.visititems(visitor)

    for name in datasets:
        if name.split("/")[-1].lower() == "img":
            return h5_file[name][:]

    raise ValueError(
        "Could not find an image dataset named 'img' in this HDF5 file."
    )


# =========================================================
# RGB IMAGE CONVERSION
# =========================================================

def create_rgb_image(image):
    # Sentinel-2 RGB bands: B4, B3, B2
    rgb = image[:, :, [3, 2, 1]].astype(np.float32)

    # Percentile stretching for visualization
    for channel in range(3):
        band = rgb[:, :, channel]

        lower = np.percentile(band, 2)
        upper = np.percentile(band, 98)

        if upper > lower:
            rgb[:, :, channel] = np.clip(
                (band - lower) / (upper - lower),
                0,
                1
            )
        else:
            rgb[:, :, channel] = 0

    return rgb


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_landslide(image, model, band_mean, band_std):
    # Normalize image using training statistics
    image_normalized = (
        image - band_mean.reshape(1, 1, 14)
    ) / (
        band_std.reshape(1, 1, 14) + 1e-7
    )

    # Add batch dimension
    input_image = np.expand_dims(
        image_normalized,
        axis=0
    )

    # Model prediction
    prediction_probability = model.predict(
        input_image,
        verbose=0
    )[0, :, :, 0]

    # Binary segmentation
    predicted_mask = (
        prediction_probability > 0.5
    ).astype(np.uint8)

    return predicted_mask, prediction_probability


# =========================================================
# TRAINING GRAPH FUNCTION
# =========================================================

def plot_training_metric(history, metric, title, ylabel):
    train_values = history.get(metric)
    val_values = history.get("val_" + metric)

    if train_values is None or val_values is None:
        st.warning(f"{title} data is not available.")
        return

    epochs = range(1, len(train_values) + 1)
    colors = get_plot_theme()

    with plt.rc_context({
        "figure.facecolor": colors["figure"],
        "axes.facecolor": colors["axes"],
        "text.color": colors["text"],
        "axes.labelcolor": colors["text"],
        "xtick.color": colors["text"],
        "ytick.color": colors["text"]
    }):
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(
            epochs,
            train_values,
            label="Training",
            linewidth=2
        )

        ax.plot(
            epochs,
            val_values,
            label="Validation",
            linewidth=2
        )

        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(ylabel)
        ax.legend()
        style_axes(ax)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# =========================================================
# CONFUSION MATRIX FUNCTION
# =========================================================

def plot_confusion_matrix(matrix, title, percentage=False):
    colors = get_plot_theme()

    with plt.rc_context({
        "figure.facecolor": colors["figure"],
        "axes.facecolor": colors["axes"],
        "text.color": colors["text"],
        "axes.labelcolor": colors["text"],
        "xtick.color": colors["text"],
        "ytick.color": colors["text"]
    }):
        fig, ax = plt.subplots(figsize=(5, 4))

        if percentage:
            image = ax.imshow(
                matrix,
                cmap="Blues",
                vmin=0,
                vmax=1
            )
        else:
            image = ax.imshow(
                matrix,
                cmap="Blues"
            )

        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Landslide", "Landslide"])
        ax.set_yticklabels(["Non-Landslide", "Landslide"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        threshold = np.nanmax(matrix) / 2 if matrix.size else 0

        for i in range(2):
            for j in range(2):
                if percentage:
                    value = f"{matrix[i, j] * 100:.2f}%"
                else:
                    value = f"{int(matrix[i, j]):,}"

                annotation_color = (
                    "white" if matrix[i, j] > threshold else "#111111"
                )

                ax.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    color=annotation_color,
                    fontweight="bold"
                )

        colorbar = fig.colorbar(image, ax=ax)
        colorbar.ax.yaxis.set_tick_params(color=colors["text"])
        plt.setp(
            colorbar.ax.get_yticklabels(),
            color=colors["text"]
        )

        style_axes(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌍 Landslide AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Landslide Prediction",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Model:** DeepLabV3+

    **Dataset:** Landslide4Sense

    **Input:** 14-channel satellite imagery

    **Task:** Binary Semantic Segmentation
    """
)

st.sidebar.caption(
    "Display mode is controlled by Streamlit. "
    "The app supports both Light and Dark themes."
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <h1>🌍 AI-Based Landslide Detection</h1>
            <p>Satellite Image Segmentation using DeepLabV3+</p>
            <p>
                An AI-powered system that identifies landslide-affected
                regions from multispectral satellite imagery.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Project Overview")

    st.write(
        """
        Landslides are natural hazards that can cause significant
        damage to infrastructure, ecosystems and human settlements.

        This project uses DeepLabV3+, a deep learning-based semantic
        segmentation architecture, to identify landslide-affected
        pixels in satellite images.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Architecture")
        st.write(
            """
            **DeepLabV3+**

            - Encoder-decoder architecture
            - Atrous Spatial Pyramid Pooling (ASPP)
            - Captures spatial and contextual information
            - Produces pixel-level segmentation masks
            """
        )

    with col2:
        st.markdown("### Dataset Information")
        st.write(
            """
            **Landslide4Sense**

            - Multispectral satellite imagery
            - 14 input channels
            - 128 × 128 image dimensions
            - Binary segmentation labels
            """
        )

    st.markdown("---")
    st.markdown("## How It Works")

    st.write(
        """
        1. Upload a satellite image in HDF5 format.
        2. The system extracts its 14 spectral channels.
        3. Input data is normalized using training statistics.
        4. DeepLabV3+ predicts landslide probabilities.
        5. A binary segmentation mask highlights the detected region.
        """
    )

    st.info(
        "Navigate to Landslide Prediction to upload an HDF5 image "
        "and generate a prediction."
    )


# =========================================================
# LANDSLIDE PREDICTION PAGE
# =========================================================

elif page == "Landslide Prediction":

    st.title("🛰️ Landslide Prediction")

    st.write(
        "Upload a Landslide4Sense-compatible HDF5 satellite image "
        "to detect landslide-affected regions."
    )

    st.warning(
        "Upload a 128 × 128 × 14 multispectral HDF5 image. "
        "Ordinary JPG and PNG images are not supported for prediction."
    )

    uploaded_file = st.file_uploader(
        "Upload Satellite Image",
        type=["h5", "hdf5"]
    )

    if uploaded_file is not None:

        try:
            with h5py.File(
                BytesIO(uploaded_file.getvalue()),
                "r"
            ) as h5_file:
                image = find_image_dataset(h5_file)

            # Remove an optional batch dimension
            image = np.squeeze(image)

            if image.shape != (128, 128, 14):
                st.error(
                    f"Unexpected image shape: {image.shape}. "
                    "Expected (128, 128, 14)."
                )
                st.stop()

            if not np.isfinite(image).all():
                st.error("The uploaded image contains invalid values.")
                st.stop()

            st.success("Satellite image uploaded successfully!")

            with st.spinner("Loading model and processing image..."):
                model = load_model()
                band_mean, band_std = load_normalization()

                predicted_mask, probability = predict_landslide(
                    image,
                    model,
                    band_mean,
                    band_std
                )

            # Calculate predicted area
            total_pixels = predicted_mask.size
            landslide_pixels = int(np.sum(predicted_mask))
            coverage = (landslide_pixels / total_pixels) * 100

            # Generate RGB image
            rgb_image = create_rgb_image(image)

            # Generate overlay
            overlay = rgb_image.copy()
            overlay[predicted_mask == 1] = (1.0, 0.1, 0.1)

            overlay_image = (
                0.55 * rgb_image +
                0.45 * overlay
            )

            # RESULT SUMMARY
            st.markdown("---")
            st.subheader("Prediction Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Landslide Pixels",
                f"{landslide_pixels:,}"
            )

            col2.metric(
                "Landslide Coverage",
                f"{coverage:.2f}%"
            )

            if landslide_pixels > 0:
                status = "Landslide Detected"
            else:
                status = "No Landslide Pixels Detected"

            col3.metric(
                "Prediction Status",
                status
            )

            # IMAGE VISUALIZATION
            st.markdown("---")
            st.subheader("Image Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Satellite Image**")
                st.image(
                    rgb_image,
                    use_container_width=True
                )

            with col2:
                st.markdown("**Predicted Segmentation Mask**")
                st.image(
                    predicted_mask * 255,
                    clamp=True,
                    use_container_width=True
                )

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("**Landslide Overlay**")
                st.image(
                    overlay_image,
                    use_container_width=True
                )

            with col4:
                st.markdown("**Prediction Probability Heatmap**")
                colors = get_plot_theme()

                with plt.rc_context({
                    "figure.facecolor": colors["figure"],
                    "axes.facecolor": colors["axes"],
                    "text.color": colors["text"]
                }):
                    fig, ax = plt.subplots(figsize=(6, 4))

                    heatmap = ax.imshow(
                        probability,
                        cmap="hot",
                        vmin=0,
                        vmax=1
                    )

                    ax.set_title("Landslide Probability")
                    ax.axis("off")

                    colorbar = fig.colorbar(
                        heatmap,
                        ax=ax,
                        fraction=0.046,
                        pad=0.04
                    )
                    colorbar.ax.yaxis.set_tick_params(
                        color=colors["text"]
                    )
                    plt.setp(
                        colorbar.ax.get_yticklabels(),
                        color=colors["text"]
                    )

                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            # DOWNLOAD PREDICTION
            st.markdown("---")
            st.subheader("Download Prediction")

            mask_image = Image.fromarray(
                predicted_mask * 255
            )

            buffer = BytesIO()
            mask_image.save(
                buffer,
                format="PNG"
            )

            st.download_button(
                label="Download Segmentation Mask",
                data=buffer.getvalue(),
                file_name="landslide_prediction.png",
                mime="image/png"
            )

        except Exception as error:
            st.error(
                f"An error occurred while processing the image: {error}"
            )


# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Evaluation results of the trained DeepLabV3+ model "
        "on the held-out test dataset."
    )

    # TEST METRICS
    st.subheader("Test Dataset Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        "98.80%"
    )

    col2.metric(
        "IoU",
        "62.11%"
    )

    col3.metric(
        "Dice Score",
        "76.63%"
    )

    col4, col5 = st.columns(2)

    col4.metric(
        "Precision",
        "74.12%"
    )

    col5.metric(
        "Recall",
        "79.32%"
    )

    st.caption(
        "These are the previously calculated test-set results. "
        "They are not calculated from the image uploaded on the prediction page."
    )

    st.info(
        "Pixel accuracy can be high when most pixels belong to the "
        "non-landslide class. IoU and Dice are also important for "
        "evaluating segmentation quality."
    )

    # CONFUSION MATRIX
    st.markdown("---")
    st.subheader("Confusion Matrix")

    try:
        cm_raw, cm_normalized = load_evaluation()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Normalized Confusion Matrix**")
            plot_confusion_matrix(
                cm_normalized,
                "Normalized Confusion Matrix",
                percentage=True
            )

        with col2:
            st.markdown("**Raw Confusion Matrix**")
            plot_confusion_matrix(
                cm_raw,
                "Raw Confusion Matrix",
                percentage=False
            )

    except Exception as error:
        st.warning(
            f"Could not load confusion matrix: {error}"
        )

    # TRAINING HISTORY
    st.markdown("---")
    st.subheader("Training and Validation Performance")

    try:
        history = load_history()

        col1, col2 = st.columns(2)

        with col1:
            plot_training_metric(
                history,
                "loss",
                "Training vs Validation Loss",
                "Loss"
            )

        with col2:
            plot_training_metric(
                history,
                "accuracy",
                "Training vs Validation Accuracy",
                "Accuracy"
            )

        col3, col4 = st.columns(2)

        with col3:
            plot_training_metric(
                history,
                "IoU",
                "Training vs Validation IoU",
                "IoU"
            )

        with col4:
            plot_training_metric(
                history,
                "precision",
                "Training vs Validation Precision",
                "Precision"
            )

        plot_training_metric(
            history,
            "recall",
            "Training vs Validation Recall",
            "Recall"
        )

    except Exception as error:
        st.warning(
            f"Could not load training history: {error}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:var(--text-color); opacity:0.7;">
        AI-Based Landslide Detection using DeepLabV3+<br>
        Built with Python, TensorFlow and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
