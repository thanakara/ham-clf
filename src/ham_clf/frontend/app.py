import io
import sys
import asyncio

from pathlib import Path

import streamlit as st

from PIL import Image
from fastapi import UploadFile

from ham_clf.backend.app import model, invoke_endpoint

backend_path = Path(__file__).parent.parent / "backend"  # adding backend to python path
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))


STATUS_OK = 200

st.set_page_config(page_title="Skin-Cancer [HAM10000]", page_icon="🔬", layout="centered")

st.title("HAM-Classifier")
st.markdown(
    """

This application uses a fine tuned MobileNetV2 model
on the HAM10000 dataset to classify skin lesions into different categories.
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"], help="Let the image be as clear as possible"
)


async def make_prediction(uploaded_file):
    """Call the FastAPI endpoint directly"""
    uploaded_file.seek(0)
    file_content = uploaded_file.read()

    file_obj = UploadFile(
        file=io.BytesIO(file_content), filename=uploaded_file.name, headers={"content-type": uploaded_file.type}
    )

    result = await invoke_endpoint(file_obj)
    return result


if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image)

    with col2:
        st.subheader("Results")

        if st.button("Predict", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    result = asyncio.run(make_prediction(uploaded_file))

                    st.success("Analysis complete")

                    st.metric(label="Class:", value=result.prediction)

                    confidence_pct = result.confidence * 100
                    st.metric(label="Confidence", value=f"{confidence_pct:.2f}%")

                    st.progress(result.confidence)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("### Classes")

    classes_info = {
        "actinic_keratoses": {
            "description": """Rough, scaly patches caused by sun damage.
            Can develop into skin cancer if untreated.""",
            "severity": "Precancerous",
            "risk": "Moderate",
        },
        "basal_cell_carcinoma": {
            "description": """Most common type of skin cancer.
            Slow-growing, rarely spreads, highly treatable when caught early.""",
            "severity": "Malignant",
            "risk": "Low metastasis risk",
        },
        "benign_keratosis": {
            "description": """Non-cancerous growths including seborrheic keratoses.
            Generally harmless but may resemble melanoma.""",
            "severity": "Benign",
            "risk": "None",
        },
        "dermatofibroma": {
            "description": """Benign fibrous nodules, often on legs.
            Firm to touch, usually harmless and common.""",
            "severity": "Benign",
            "risk": "None",
        },
        "melanoma": {
            "description": """Most dangerous skin cancer.
            Can spread rapidly. Early detection is critical for treatment success.""",
            "severity": "Malignant",
            "risk": "High metastasis risk",
        },
        "melanocytic_nevi": {
            "description": """Common benign moles.
            Most are harmless, but changes in size, color, or shape warrant evaluation.""",
            "severity": "Benign",
            "risk": "Monitor for changes",
        },
        "vascular_lesions": {
            "description": """Blood vessel growths like cherry angiomas or hemangiomas. Typically benign.""",
            "severity": "Benign",
            "risk": "None",
        },
    }

    for class_name, info in classes_info.items():
        with st.expander(f"**{class_name}**"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Severity:** {info['severity']}")
            st.markdown(f"**Risk Level:** {info['risk']}")

    st.divider()

    st.subheader("Backend Status")
    try:
        if model is not None:
            st.success("Online")
        else:
            st.error("Offline")
    except Exception as e:
        st.error(f"Backend Error: {str(e)}")

col_disclaimer, col_technical = st.columns([1, 1])

with col_disclaimer:
    st.subheader("⚕️ Medical Disclaimer")
    st.markdown("""
    This tool is for educational purposes only and should **not** be used as a
    substitute for professional medical advice, diagnosis, or treatment.

    Always consult a qualified dermatologist for any skin concerns.
    """)

with col_technical:
    st.subheader("🔧 Technical Details")
    st.markdown("""
    - **Model**: MobileNetV2 (Fine-tuned)
    - **Dataset**: HAM10000
    - **Backend**: FastAPI & TensorFlow
    - **Framework**: TensorFlow 2.20
    """)

st.divider()
st.caption("•/thK•")
