import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile

IMG_SIZE = 155
GENUINE_THRESHOLD = 0.50
DIFFERENT_PERSON_THRESHOLD = 1.00

st.set_page_config(
    page_title="Cyber-Forensic Signature AI",
    page_icon="🕵️‍♀️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background:
    radial-gradient(circle at top left, #0ea5e9 0%, transparent 25%),
    radial-gradient(circle at bottom right, #7c3aed 0%, transparent 25%),
    linear-gradient(135deg, #020617, #0f172a, #111827);
    color: white;
}

.hero {
    padding: 35px;
    border-radius: 30px;
    text-align: center;
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(56, 189, 248, 0.35);
    box-shadow: 0 0 45px rgba(56, 189, 248, 0.35);
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 52px;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #cbd5e1;
    font-size: 19px;
}

.glass-card {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 24px;
    padding: 25px;
    box-shadow: 0 0 30px rgba(14, 165, 233, 0.22);
}

.result-genuine {
    background: rgba(34, 197, 94, 0.18);
    border: 2px solid #22c55e;
    color: #bbf7d0;
}

.result-forged {
    background: rgba(234, 179, 8, 0.18);
    border: 2px solid #eab308;
    color: #fef3c7;
}

.result-different {
    background: rgba(239, 68, 68, 0.18);
    border: 2px solid #ef4444;
    color: #fecaca;
}

.result-box {
    padding: 30px;
    border-radius: 25px;
    text-align: center;
    font-size: 32px;
    font-weight: 900;
    box-shadow: 0 0 30px rgba(255,255,255,0.15);
    margin-top: 25px;
}

.metric-box {
    background: rgba(30, 41, 59, 0.88);
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 25px;
    border-radius: 22px;
    text-align: center;
    font-size: 22px;
    box-shadow: inset 0 0 18px rgba(56, 189, 248, 0.18);
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 45px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True) + 1e-10)

@st.cache_resource
def load_signature_model():
    return tf.keras.models.load_model(
        "signature_siamese_model.keras",
        custom_objects={"euclidean_distance": euclidean_distance},
        compile=False,
        safe_mode=False
    )

model = load_signature_model()

def preprocess_signature(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)

    return img

def generate_heatmap(path1, path2):
    img1 = preprocess_signature(path1).squeeze()
    img2 = preprocess_signature(path2).squeeze()

    diff = cv2.absdiff(
        (img1 * 255).astype("uint8"),
        (img2 * 255).astype("uint8")
    )

    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap

def verify_signature(path1, path2):
    img1 = np.expand_dims(preprocess_signature(path1), axis=0)
    img2 = np.expand_dims(preprocess_signature(path2), axis=0)

    distance = model.predict([img1, img2], verbose=0)[0][0]
    similarity = max(0, 100 - distance * 100)

    if distance < GENUINE_THRESHOLD:
        result = "GENUINE SIGNATURE"
    elif distance < DIFFERENT_PERSON_THRESHOLD:
        result = "SUSPICIOUS / FORGED"
    else:
        result = "DIFFERENT PERSON"

    return result, distance, similarity

st.markdown("""
<div class="hero">
    <h1>🕵️‍♀️ Cyber-Forensic Signature AI</h1>
    <p>Writer-independent signature verification using Siamese Neural Network + forensic heatmap analysis</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📄 Original Evidence")
    original_file = st.file_uploader(
        "Upload original/reference signature",
        type=["png", "jpg", "jpeg"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔍 Questioned Evidence")
    questioned_file = st.file_uploader(
        "Upload questioned/test signature",
        type=["png", "jpg", "jpeg"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

if original_file and questioned_file:
    temp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp1.write(original_file.read())
    temp1.close()

    temp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp2.write(questioned_file.read())
    temp2.close()

    st.markdown("## 🧾 Uploaded Signature Evidence")

    img1_col, img2_col = st.columns(2)

    with img1_col:
        st.image(temp1.name, caption="Original Signature", use_container_width=True)

    with img2_col:
        st.image(temp2.name, caption="Questioned Signature", use_container_width=True)

    if st.button("🚀 RUN FORENSIC VERIFICATION", use_container_width=True):
        result, distance, similarity = verify_signature(temp1.name, temp2.name)

        if result == "GENUINE SIGNATURE":
            css_class = "result-genuine"
            icon = "✅"
        elif result == "SUSPICIOUS / FORGED":
            css_class = "result-forged"
            icon = "⚠️"
        else:
            css_class = "result-different"
            icon = "❌"

        st.markdown(
            f"""
            <div class="result-box {css_class}">
                {icon} FINAL VERDICT: {result}
            </div>
            """,
            unsafe_allow_html=True
        )

        m1, m2 = st.columns(2)

        with m1:
            st.markdown(
                f"""
                <div class="metric-box">
                    📏 Distance Score<br>
                    <b>{distance:.4f}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-box">
                    📊 Similarity Score<br>
                    <b>{similarity:.2f}%</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("## 🔥 Forensic Heatmap")

        heatmap = generate_heatmap(temp1.name, temp2.name)

        st.image(
            heatmap,
            caption="Red/yellow regions show stronger mismatch zones",
            use_container_width=True
        )

        st.info("This heatmap highlights suspicious visual differences between both signatures.")

st.markdown(
    '<div class="footer">Cyber-Forensic Signature Forgery Detection | Built with Siamese Neural Network</div>',
    unsafe_allow_html=True
)