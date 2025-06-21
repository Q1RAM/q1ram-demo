import streamlit as st

# Page config
st.set_page_config(page_title="Q1RAM App Suite", layout="wide")

# Centered logo and title
col1, col2, col3 = st.columns([3,1,3])
# with col1:
#     st.markdown("try")
with col2:
    st.markdown("<h2 style='text-align: center; margin-bottom: -50px;margin-top:-50px'>Try</h2>", unsafe_allow_html=True)
    st.image("q1ram_logo.jpg",width=200)
# Intro text
st.markdown("""
<div style='text-align: center; font-size: 18px;'>
   <strong> for free </strong> in different quantum computing technologies through <strong>Qiskit</strong> platform.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# App overview
st.markdown("""
# Try QRAM in Quantum AI
""")
st.markdown("""
### Transfer your  dataset from classical RAM to QRAM
Upload your dataset as an Excel sheet file. Then, you can use this demo to load the dataset from classical RAM to QRAM through our developed quantum gateway system. Therefore you can proceeed through this demo to write the dataset from classical RAM to QRAM, and to read the quantum data from QRAM using quantum computer simulator.
""")
st.page_link("pages/Excel_Demo.py", label=" # Open Demo Page")

st.markdown("---")
st.markdown("""
<div style='font-size: 16px; line-height: 1.6;'>
    <h1>Try QRAM in Quantum Image Processing</h1>
    <h3 style="margin-left:50px;"><strong>Store & Retrieve Quantum Images</strong></h3>
    <ul style="margin-left:70px;">
        <li>Encode a grayscale image using <strong>FRQI</strong> or <strong>NEQR</strong>.</li>
        <li>Store it in QRAM at a specific address.</li>
        <li>Visualize the quantum-stored image using simulator results.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
# st.page_link("pages/Image_Storage_Demo.py", label=" # Open Demo Page")
col1, col2 = st.columns([1, 16])  # Left spacer, content column

with col2:
    st.page_link("pages/Image_Storage_Demo.py", label="📷 Open Demo Page")

# st.markdown("---")
# st.markdown("""
# # Try QRAM in Quantum Networking Systems (coming soon)
# """)
# st.markdown("---")
# st.markdown("""
# # Try QRAM in Quantum Cybersecurity Systems (coming soon)
# """)
# st.markdown("---")
# st.markdown("""
# # Try QRAM in Quantum Sensing Systems (coming soon)
# """)
# st.markdown("---")
# st.markdown("""
# # Try QRAM in Quantum Chemistry & Drug Discovery Systems (coming soon)
# """)
# st.markdown("---")
# st.markdown("""
# # Try QRAM in Quantum Drug Discovery Systems (coming soon)
# """)
# st.markdown("---")
# st.markdown("""
# # Try QRAM in Quantum Optimization Systems (coming soon)
# """)
def coming_soon(title, emoji, color):
    st.markdown("---")
    st.markdown(f"### {emoji} {title} <span style='color:{color};'>(Coming Soon)</span>", unsafe_allow_html=True)

modules = [
    ("QRAM in Quantum Networking Systems", "🌐", "darkorange"),
    ("QRAM in Quantum Cybersecurity Systems", "🔐", "darkorange"),
    ("QRAM in Quantum Sensing Systems", "🧭", "darkorange"),
    ("QRAM in Quantum Chemistry", "⚗️", "darkorange"),
    ("QRAM in Quantum Drug Discovery", "💊", "darkorange"),
    ("QRAM in Quantum Optimization Systems", "📦", "darkorange"),
]

for title, emoji, color in modules:
    coming_soon(title, emoji, color)
