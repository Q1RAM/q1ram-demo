import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from Q1RAM import Q1RAM
from QIP import FRQI_encoding, decode_frqi_image_aer,group_dict_by_prefix

# --- Helper Functions ---
def encode_image(source_image):
  h,w=source_image.shape
  n_w=int(np.ceil(np.log2(w)))
  n_h=int(np.ceil(np.log2(h)))
  qc= QuantumCircuit()
  qr_position,qr_color=FRQI_encoding(qc,source_image,apply_h=True)

  qc_tc=qc.decompose().decompose().decompose()
  qc_tc.save_statevector()
  
  simulator=AerSimulator(method="statevector")
  zero_threshold=0.05/(2**(2*len(qr_position)+len(qr_color)))
  statevector=simulator.run(qc_tc).result().get_statevector(qc_tc)
  probabilities=statevector.probabilities_dict()
  result_image=decode_frqi_image_aer(probabilities,n_w)
  return result_image

def write_image(source_image):
    w, h = source_image.shape
    n_w = int(np.ceil(np.log2(w)))
    n_h = int(np.ceil(np.log2(h)))

    qc = QuantumCircuit()
    qr_position, qr_color = FRQI_encoding(qc, source_image, apply_h=True)

    qr_address_bus = QuantumRegister(1, name="address")
    qc.add_register(qr_address_bus)
    qr_data_bus = [*qr_position, *qr_color]

    qram = Q1RAM(len(qr_address_bus), len(qr_data_bus), qc, qr_address_bus=qr_address_bus, qr_data_bus=qr_data_bus)
    qram.apply_write()
    # print(f"total qubits:{qc.num_qubits}")
    # print(qc.draw("text",fold=-1))
    qc_tc = qc.decompose().decompose().decompose()
    qc_tc.save_statevector()
    # print("state vector saved")
    simulator = AerSimulator(method="statevector")
    result= simulator.run(qc_tc).result()
    
    statevector = result.get_statevector(qc_tc)
    probabilities =statevector.probabilities_dict(qram.qr_data_register_index+qram.qr_address_register_index )
    
    zero_threshold=0.05/(2**(2*len(qr_position)+len(qr_color)))
    correction_factor = 2 ** (2 * len(qr_address_bus))
    probabilities = {k: v * correction_factor for k, v in probabilities.items() if v > zero_threshold}

    images_probs = group_dict_by_prefix(probabilities, len(qr_address_bus))
    return {k: decode_frqi_image_aer(probs, n_w) for k, probs in images_probs.items()}

# --- Streamlit UI with Session State ---
st.set_page_config(page_title="QRAM Image Demo", layout="centered")
# st.title("Quantum Image Processing with QRAM")

# Init session state
if "source_image" not in st.session_state:
    st.session_state.source_image = None
if "encoded_image" not in st.session_state:
    st.session_state.encoded_image = None
if "qram_images" not in st.session_state:
    st.session_state.qram_images = None

# cols= st.columns(3)
# with cols[1]:
st.image("q1ram_logo.jpg", width=150)
st.title("Image Demo")
st.write("This demo showcases how to encode a grayscale image and store it in QRAM. "
         "You can upload a grayscale image, encode it, and then write it to QRAM for further processing.")

# Step 1: Upload image
st.subheader("Step 1: Upload a grayscale image into classical RAM")
st.image("image_demo_files//step1.png")
code_snippet = """
from QIP.utils import load_image

source_image = load_image(path,target_size=(32, 32))
"""

st.code(code_snippet, language="python",line_numbers=True)
uploaded_file = st.file_uploader("Step 1: Upload grayscale image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    from PIL import Image
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((32, 32))  # Resize to 32x32 pixels
    image_np = np.array(image)
    st.session_state.source_image = image_np
    col1,col2,col3 =st.columns(3)

    with col2:
        st.image(image_np, caption="Uploaded Image")

# Step 2: Encode Image
st.subheader("Step 2: Quantum image Encoding System")
st.markdown("This step is required to load the image from classical RAM into the quantum data bus |DR>")
code_snippet = """
from QIP.representations import FRQIRepresentation

frqi_image = FRQIRepresentation.from_image(source_image,target_size=(32, 32))
                                .apply_encoding()
                                .measure()
"""

st.code(code_snippet, language="python",line_numbers=True)
st.image("image_demo_files//step2.png")
if st.session_state.source_image is not None:
    if st.button("Step 2: Encode Image"):
        st.write(st.session_state.source_image.shape)
        st.session_state.encoded_image = encode_image(st.session_state.source_image)

    if st.session_state.encoded_image is not None:
        st.subheader("Decoded Image")
        col1,col2,col3 =st.columns([1,1,1])
        with col2:
            # st.image(st.session_state.encoded_image)
            fig = plt.figure(figsize=(1, 1))
            fig, ax = plt.subplots()
            ax.imshow(st.session_state.encoded_image, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)

# Step 3: Write to QRAM
st.header("Step 3: Write to QRAM and Decode")
st.image("image_demo_files//step3.png")
code_snippet = """
from Q1RAM import QRAM
qr_address_bus= QuantumRegister(1,name="qr_address")
qr_data_bus=[*frqi_image.qr_position,*frqi_image.qr_color]
qram= QRAM(circuit=qc,address_bus=qr_address_bus, data_bus=qr_data_bus)
qram.apply_write()
qram.measure_internal()
"""

st.code(code_snippet, language="python",line_numbers=True)
if st.session_state.source_image is not None:
    if st.button("Step 3: Write to QRAM and Decode"):
        st.session_state.qram_images = write_image(st.session_state.source_image)

    if st.session_state.qram_images:
        st.subheader("Decoded Images from QRAM")
        num_images = len(st.session_state.qram_images)
        fig = plt.figure(figsize=(2 * num_images, 2))
        for i, (k, v) in enumerate(st.session_state.qram_images.items()):
            plt.subplot(1, num_images, i + 1)
            plt.title(f"Image at address {k}")
            plt.imshow(v, cmap='gray')
            plt.axis('off')
        st.pyplot(fig)
st.markdown("---")
col1,col2,col3= st.columns([1,3,1])
with col2:
    st.link_button("👉 Subscribe here to integrate QRAM in your system","https://lab.q1ram.com")
# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from qiskit import QuantumCircuit, QuantumRegister, transpile,ClassicalRegister
# from qiskit_aer import AerSimulator

# from Q1RAM import Q1RAM
# from QIP import (
#     FRQI_encoding,
#     decode_frqi_image_aer,
#     encode_color_in_position_with_UCRY,
#     reconstruct_neqr_image,
#     group_dict_by_prefix
# )

# # ---- Configurable App Settings ----
# st.set_page_config(page_title="QRAM Image Demo", layout="centered")
# st.image("q1ram_logo.jpg", width=150)
# st.title("QRAM Image Demo")

# # ---- Sidebar Settings ----
# st.sidebar.header("Configuration")
# encoding_method = st.sidebar.selectbox("Encoding Method", ["FRQI", "NEQR"])
# image_size = st.sidebar.selectbox("Resize Image To", [8, 16, 32, 64], index=2)
# color_bit_depth = st.sidebar.slider("Color Bit Depth", 1, 8, value=4)
# num_shots = st.sidebar.number_input("Number of Shots", min_value=1, value=10000, step=100)

# # ---- Session State Init ----
# if "source_image" not in st.session_state:
#     st.session_state.source_image = None
# if "encoded_image" not in st.session_state:
#     st.session_state.encoded_image = None
# if "qram_images" not in st.session_state:
#     st.session_state.qram_images = None

# # ---- Helper: Encode Image ----
# def encode_image(source_image, method, color_bit_depth,num_shots=num_shots):
#     h, w = source_image.shape
#     n_w = int(np.ceil(np.log2(w)))

#     qc = QuantumCircuit()

#     if method == "FRQI":
#         qr_position, qr_color = FRQI_encoding(qc, source_image, apply_h=True)
#     elif method == "NEQR":
#         qr_position, qr_color = encode_color_in_position_with_UCRY(source_image, qc, q=color_bit_depth, apply_h=True)
#     else:
#         raise ValueError("Invalid method.")
    
#     cr_position= ClassicalRegister(len(qr_position),name="pos")
#     cr_color= ClassicalRegister(len(qr_color),name="color")
#     qc.add_register(cr_position,cr_color)
#     qc.measure(qr_position,cr_position)
#     qc.measure(qr_color,cr_color)
#     simulator = AerSimulator()
#     qc_tc=transpile(qc,simulator,optimization_level=3)
#     print(f"num shots method:{num_shots}")
#     counts = simulator.run(qc_tc,shots=num_shots).result().get_counts()
#     print(counts)
#     probabilities={k:v/num_shots for k,v in counts.items()}


#     if method == "FRQI":
#         return decode_frqi_image_aer(probabilities, n_w)
#     elif method == "NEQR":
#         n_h = int(np.ceil(np.log2(h)))
#         return reconstruct_neqr_image(probabilities, n_h, n_w, color_bit_depth,filter_probs=True)

# # ---- Helper: Write to QRAM ----
# def write_image(source_image, method="FRQI", color_bit_depth=4, num_shots=1024):
#     h, w = source_image.shape
#     n_w = int(np.ceil(np.log2(w)))
#     n_h = int(np.ceil(np.log2(h)))

#     qc = QuantumCircuit()

#     if method == "FRQI":
#         qr_position, qr_color = FRQI_encoding(qc, source_image, apply_h=True)
#     elif method == "NEQR":
#         qr_position, qr_color = encode_color_in_position_with_UCRY(source_image, qc, q=color_bit_depth, apply_h=True)
#     else:
#         raise ValueError("Unsupported method.")

#     qr_address_bus = QuantumRegister(1, name="address")
#     qc.add_register(qr_address_bus)
#     qr_data_bus = [*qr_position, *qr_color]

#     qram = Q1RAM(len(qr_address_bus), len(qr_data_bus), qc, qr_address_bus=qr_address_bus, qr_data_bus=qr_data_bus)
#     qram.apply_write()

#     qram.Measure_Internal_Data()
#     simulator = AerSimulator(method="matrix_product_state")
#     qc_tc=transpile(qc,simulator,optimization_level=3)
#     counts = simulator.run(qc_tc,shots=num_shots).result().get_counts()
#     probabilities={k:v/num_shots for k,v in counts.items()}

#     # indices = qram.qr_data_register_index + qram.qr_address_register_index
#     # probabilities = statevector.probabilities_dict(indices)

#     zero_threshold = 0.05 / (2 ** (2 * len(qr_position) + len(qr_color)))
#     correction_factor = 2 ** (2 * len(qr_address_bus))
#     filtered_probs = {k: v * correction_factor for k, v in probabilities.items() if v > zero_threshold}

#     images_probs = group_dict_by_prefix(filtered_probs, len(qr_address_bus))

#     if method == "FRQI":
#         return {
#             address: decode_frqi_image_aer(probs, n_w)
#             for address, probs in images_probs.items()
#         }
#     else:  # NEQR
#         return {
#             address: reconstruct_neqr_image(probs, n_h, n_w, color_bit_depth)
#             for address, probs in images_probs.items()
#         }

# # ---- Step 1: Upload Image ----
# st.subheader("Step 1: Upload a grayscale image into classical RAM")
# st.image("image_demo_files/step1.png")

# uploaded_file = st.file_uploader("Upload grayscale image", type=["png", "jpg", "jpeg"])
# if uploaded_file:
#     image = Image.open(uploaded_file).convert("L")
#     image = image.resize((image_size, image_size))
#     image_np = np.array(image)
#     st.session_state.source_image = image_np
#     st.image(image_np, caption="Uploaded Image", use_container_width=False, width=200)

# # ---- Step 2: Encode Image ----
# st.subheader("Step 2: Quantum image Encoding System")
# st.markdown("This step is required to load the image from classical RAM into the quantum data bus |DR>")
# st.image("image_demo_files/step2.png")

# if st.session_state.source_image is not None:
#     if st.button("Run Encoding"):
#         st.session_state.encoded_image = encode_image(
#             st.session_state.source_image,
#             method=encoding_method,
#             color_bit_depth=color_bit_depth
#         )

#     if st.session_state.encoded_image is not None:
#         st.image(st.session_state.encoded_image, caption="Decoded Image", clamp=True, use_container_width=False, width=200)

# # ---- Step 3: QRAM Write ----
# st.subheader("Step 3: Write to QRAM and Decode")
# st.image("image_demo_files/step3.png")

# if st.session_state.source_image is not None:
#     if st.button("Write to QRAM"):
#         st.session_state.qram_images = write_image(
#             st.session_state.source_image,
#             method=encoding_method,
#             color_bit_depth=color_bit_depth,
#             num_shots=num_shots
#         )

#     if st.session_state.qram_images:
#         st.subheader("Decoded Images from QRAM")
#         num_images = len(st.session_state.qram_images)
#         fig = plt.figure(figsize=(2 * num_images, 2))
#         for i, (k, v) in enumerate(st.session_state.qram_images.items()):
#             plt.subplot(1, num_images, i + 1)
#             plt.title(f"Address {k}")
#             plt.imshow(v, cmap='gray', vmin=0, vmax=255 if color_bit_depth == 8 else 2**color_bit_depth - 1)
#             plt.axis('off')
#         st.pyplot(fig)
# st.markdown("---")
# col1,col2,col3= st.columns([1,3,1])
# with col2:
#     st.button("👉 Subscribe here to integrate QRAM in your system")