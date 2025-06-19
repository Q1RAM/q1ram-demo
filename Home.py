import streamlit as st

# Page config
st.set_page_config(page_title="Q1RAM App Suite", layout="centered")

# Centered logo and title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("q1ram_logo.jpg", width=250)
    st.markdown("<h1 style='text-align: center; margin-top: -10px;'>Quantum App Suite</h1>", unsafe_allow_html=True)

# Intro text
st.markdown("""
<div style='text-align: center; font-size: 18px;'>
    Explore the intersection of classical and quantum data processing using <strong>Qiskit</strong> and <strong>Q1RAM</strong>.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# App overview
st.subheader("🚀 Available Demos")

st.markdown("""
### 📊 Classical Data QRAM Demo
Upload an Excel sheet, encode classical values into quantum states using the **Quantum Gateway**, and interactively write/read from QRAM.

### 📷 Quantum Image Demo
Upload a grayscale image, encode it, and simulate QRAM-based storage and retrieval using quantum simulations.
""")

# Instruction
st.markdown("---")
st.info("📂 Use the **sidebar** to navigate between demos.")
