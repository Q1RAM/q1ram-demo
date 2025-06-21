import streamlit as st

# Page config
st.set_page_config(page_title="Q1RAM App Suite", layout="centered")

# Centered logo and title
col1, col2, col3 = st.columns([1, 2, 1])
# with col1:
#     st.markdown("try")
with col2:
    st.markdown("<h2 style='text-align: center; margin-bottom: -50px;margin-top:-50px'>Try</h2>", unsafe_allow_html=True)
    st.image("q1ram_logo.jpg", width=200)
# Intro text
st.markdown("""
<div style='text-align: center; font-size: 18px;'>
   <strong> for free </strong> in different quantum computing technologies through <strong>Qiskit</strong> platform.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# App overview
st.subheader("🚀 Available Demos")
st.markdown("""
# Try QRAM in Quantum AI
""")
st.markdown("""
### Transfer your  dataset from classical RAM to QRAM
Upload your dataset as an Excel sheet file. Then, you can use this demo to load the dataset from classical RAM to QRAM through our developed quantum gateway system. Therefore you can proceeed through this demo to write the dataset from classical RAM. to QRAM, and to read the quantum data from QRAM using quantum computer simulator.
""")

# Instruction
st.markdown("---")
st.markdown("""
# Try QRAM in Quantum image processing
""")
st.markdown("---")
st.markdown("""
# Try QRAM in Quantum Networking Systems (coming soon)
""")
st.markdown("---")
st.markdown("""
# Try QRAM in Quantum Cybersecurity Systems (coming soon)
""")
st.markdown("---")
st.markdown("""
# Try QRAM in Quantum Sensing Systems (coming soon)
""")
st.markdown("---")
st.markdown("""
# Try QRAM in Quantum Chemistry & Drug Discovery Systems (coming soon)
""")
st.markdown("---")
st.markdown("""
# Try QRAM in Quantum Drug Discovery Systems (coming soon)
""")
st.markdown("---")
st.markdown("""
# Try QRAM in Quantum Optimization Systems (coming soon)
""")