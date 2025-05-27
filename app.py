import streamlit as st
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.visualization import matplotlib as qiskit_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ClassicalQuantumGateway import *
from Q1RAM import *
import os

# --- Session State Initialization ---
if "step" not in st.session_state:
    st.session_state.step = 1
if "excel_data" not in st.session_state:
    st.session_state.excel_data = None
if "rows_values" not in st.session_state:
    st.session_state.rows_values = []
if "cols" not in st.session_state:
    st.session_state.cols = []
if "col_widths" not in st.session_state:
    st.session_state.col_widths = []
if "encode_output" not in st.session_state:
    st.session_state.encode_output = None
if "write_output" not in st.session_state:
    st.session_state.write_output = None
if "read_output" not in st.session_state:
    st.session_state.read_output = None
if "addresses" not in st.session_state:
    st.session_state.addresses = []

def plot_results_st(counts, n, m, cols, col_widths, bar_color='blue'):
    address_qubits = n
    data_qubits = m
    sorted_counts = dict(sorted(counts.items(), key=lambda item: int(item[0][-address_qubits:], 2)))

    address_values = []
    counts_values = []
    for address, count in sorted_counts.items():
        address_values.append(address)
        counts_values.append(count)

    x_labels = []
    for address in address_values:
        formatted_address_data = add_hyphens_to_bitstring(address[:data_qubits], col_widths)
        x_labels.append(f"{address[-address_qubits:]}:({formatted_address_data})")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x_labels, counts_values, color=bar_color)

    for i, bar in enumerate(bars):
        if counts_values[i] == 0:
            bar.set_color('lightgray')

    address_counts = {}
    for address in address_values:
        address_counts[address[-address_qubits:]] = address_counts.get(address[-address_qubits:], 0) + 1
    for i, bar in enumerate(bars):
        if address_counts[address_values[i][-address_qubits:]] > 1 and int(address_values[i][:data_qubits]) == 0:
            bar.set_color('lightgray')

    light_gray_patch = plt.Rectangle((0, 0), 1, 1, fc="lightgray")
    ax.legend([light_gray_patch], ["Empty cell value"])

    ax.set_xlabel(f"Address:({'- '.join(cols)})")
    ax.set_ylabel("Counts")
    ax.set_title("QRAM Read Results")
    ax.set_xticklabels(x_labels, rotation=90)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

def load_excel(file):
    st.session_state.excel_data = pd.read_excel(file)
    row_count = len(st.session_state.excel_data)
    st.session_state.rows_values, st.session_state.cols, st.session_state.col_widths = process_excel_file(file)
    return row_count

def indicies_to_statevector_st(indicies):
  n_max = max(indicies)
  


  n_bits = np.ceil(np.log2(len(st.session_state.rows_values))).astype(int)
  
  total_bits = n_bits 
  state_vector_size = 2**total_bits
  state_vector = np.zeros(state_vector_size, dtype=complex) # Use complex for state vectors

  num_data_points = len(indicies)
  amplitude = 1.0 / np.sqrt(num_data_points)

  for index in indicies:
    

    # Assign amplitude to the corresponding index
    state_vector[index] = amplitude

  return state_vector

def encode_data(shots=1024):
    rows_values = st.session_state.rows_values
    cols = st.session_state.cols
    col_widths = st.session_state.col_widths
    address_qubits = int(np.ceil(np.log2(len(rows_values))))
    data_qubits = int(np.ceil(np.log2(np.max(rows_values))))
    classical_data = list(zip(range(len(rows_values)), rows_values))
    qc = QuantumCircuit()
    qr_AR, qr_DR, qr_Ar, qr_Cd, cb1, cb2, qr_tof_ancillae = encode_classical_data(qc, classical_data)
    cr_address = ClassicalRegister(address_qubits)
    cr_data = ClassicalRegister(data_qubits)
    cr_read_flag = ClassicalRegister(1)
    qc.add_register(cr_address, cr_data)
    qc.measure(qr_AR, cr_address)
    qc.measure(qr_DR, cr_data)
    counts = simulate_circuit(qc,shots=shots)
    st.session_state.encode_output = (counts, address_qubits, data_qubits, cols, col_widths)

def write_qram(shots=1024):
    rows_values = st.session_state.rows_values
    cols = st.session_state.cols
    col_widths = st.session_state.col_widths
    address_qubits = int(np.ceil(np.log2(len(rows_values))))
    data_qubits = int(np.ceil(np.log2(np.max(rows_values))))
    classical_data = list(zip(range(len(rows_values)), rows_values))
    qc = QuantumCircuit()
    qr_AR, qr_DR, qr_Ar, qr_Cd, cb1, cb2, qr_tof_ancillae = encode_classical_data(qc, classical_data)
    qram = Q1RAM(address_qubits, data_qubits, qc=qc, qr_address_bus=qr_AR, qr_data_bus=qr_DR)
    qram.apply_write()
    qram.Measure_Internal_Data()
    counts = simulate_circuit(qc,shots=shots)
    st.session_state.write_output = (counts, address_qubits, data_qubits, cols, col_widths)

def read_qram(addresses,shots=1024):
    rows_values = st.session_state.rows_values
    cols = st.session_state.cols
    col_widths = st.session_state.col_widths
    address_qubits = int(np.ceil(np.log2(len(rows_values))))
    data_qubits = int(np.ceil(np.log2(np.max(rows_values))))
    classical_data = list(zip(range(len(rows_values)), rows_values))
    qc = QuantumCircuit()
    qr_AR, qr_DR, qr_Ar, qr_Cd, cb1, cb2, qr_tof_ancillae = encode_classical_data(qc, classical_data)
    qram = Q1RAM(address_qubits, data_qubits, qc=qc, qr_address_bus=qr_AR, qr_data_bus=qr_DR)
    qram.apply_write()
    qram.qc.reset(qr_AR)
    qram.qc.reset(qr_DR)
    qram.qc.reset(qr_Ar)
    qram.qc.reset(qr_Cd)
    qram.qc.reset(cb1)
    qram.qc.reset(cb2)
    qram.qc.reset(qr_tof_ancillae)
    if len(addresses) > 0:
        qram.qc.prepare_state(indicies_to_statevector_st(addresses), qr_AR)
        qram.apply_read()
    else:
        qram.ReadAll()
    qram.Measure()
    counts = simulate_circuit(qc,shots=shots)
    st.session_state.read_output = (counts, address_qubits, data_qubits, cols, col_widths)

def start_over():
    # Delete the uploaded Excel file if it was saved to disk
    if "uploaded_file_path" in st.session_state:
        try:
            if os.path.exists(st.session_state.uploaded_file_path):
                os.remove(st.session_state.uploaded_file_path)
        except Exception as e:
            st.warning(f"Could not delete file: {e}")
        del st.session_state["uploaded_file_path"]
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 1
    st.rerun()
st.set_page_config(page_title="Q1RAM Demo", page_icon=":guardsman:", layout="wide")

# Insert a static image and title in the same row at the top
col1, col2, col3 = st.columns([1, 3, 1])

with col2: # Place the image in the middle column
    st.image("qram.png", use_container_width=True)

# st.image("qram.png", width=300)  # Adjust width as needed
# with header_col2:
# st.header("Demo")

with st.expander("About this demo", expanded=True):
    st.write("""
    This demo showcases the Quantum Gateway System and QRAM (Quantum Random Access Memory) using a classical datasetin the following format:
             
             feature1| feature2| ... | featureN| class

             value1 | value2 | ... | valueN | class_value
             
    The Quantum Gateway System encodes classical data into quantum states, which are then written to and read from QRAM.
    The demo is divided into four steps:
    1. Upload an Excel file containing the dataset.
    2. Apply the Quantum Gateway System to encode the data into quantum states.
    3. Write the encoded quantum states into QRAM.
    4. Read the quantum states from QRAM.
    """)
    st.write(""" This demo is built using Streamlit and Qiskit, and it allows you to visualize the results of each step in the process.""")
# Set up the page configuration
# Set up the matplotlib backend for Streamlit
# Set the title of the app
# Set the default number of shots for quantum circuit simulation
if "num_shots" not in st.session_state:
    st.session_state.num_shots = 1024
# Add a number input for the number of shots
st.sidebar.header("Settings")
st.sidebar.write("Adjust the number of shots for quantum circuit simulation:")
shots = st.sidebar.number_input("Number of shots (for quantum circuit simulation):",min_value=1,max_value=100000,value=1024,step=1,key="num_shots")

# Step 1: Upload Excel
st.header("Step 1: Upload Excel File")
st.image("./step1_loading_dataset.png", use_container_width=True)
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"], disabled=st.session_state.step > 1)
if uploaded_file is not None and st.session_state.step == 1:
    # Optionally save the uploaded file to disk for deletion later
    temp_path = "uploaded_excel_temp.xlsx"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.uploaded_file_path = temp_path
    row_count = load_excel(temp_path)
    st.success(f"Loaded {row_count} rows from Excel file.")
    st.session_state.step = 2

if st.session_state.excel_data is not None:
    st.write("Preview:", st.session_state.excel_data)

# Step 2: Encode classical data
st.header("Step 2: Apply the Quantum Gateway System")
st.image("./step2.png", use_container_width=True)
encode_disabled = st.session_state.step != 2 or st.session_state.get("encode_loading", False)


if st.button("Simulate Applying Quantum Gateway", disabled=encode_disabled):
    st.session_state.encode_loading = True
    with st.spinner("Applying Quantum Gateway..."):
        try:
            encode_data(shots=st.session_state.get("num_shots", 1024))
            st.session_state.step = 3
        except Exception as e:
            st.error(f"Error during Gateway: {e}")
    st.session_state.encode_loading = False

if st.session_state.encode_output is not None:
    counts, address_qubits, data_qubits, cols, col_widths = st.session_state.encode_output
    st.subheader("Quantum Gateway Result")
    plot_results_st(counts, address_qubits, data_qubits, cols, col_widths, bar_color='blue')

# Step 3: Write into QRAM
st.header("Step 3: Write into QRAM")
st.image("./step3_write.png", use_container_width=True)
write_disabled = st.session_state.step != 3 or st.session_state.get("write_loading", False)
if st.button("Write to QRAM", disabled=write_disabled):
    st.session_state.write_loading = True
    with st.spinner("Writing to QRAM..."):
        write_qram(shots=st.session_state.get("num_shots", 1024))
        st.session_state.step = 4
    st.session_state.write_loading = False

if st.session_state.write_output is not None:
    counts, address_qubits, data_qubits, cols, col_widths = st.session_state.write_output
    st.subheader("Write Result")
    plot_results_st(counts, address_qubits, data_qubits, cols, col_widths, bar_color='red')

# Step 4: Read from QRAM
st.header("Step 4: Read from QRAM")
st.image("./step4_read.png", use_container_width=True)
read_disabled = st.session_state.step != 4 or st.session_state.get("read_loading", False)

row_count = len(st.session_state.rows_values) if st.session_state.rows_values else 0
address_options = list(range(row_count))
selected_addresses = st.multiselect(
    f"Select address(es) to read (0 to {row_count-1}). Leave empty to read all:",
    options=address_options,
    disabled=read_disabled
)
if st.button("Read from QRAM", disabled=read_disabled):
    st.session_state.read_loading = True
    with st.spinner("Reading from QRAM..."):
        addresses = selected_addresses if selected_addresses else []
        st.session_state.addresses = addresses
        read_qram(addresses,shots=st.session_state.get("num_shots", 1024))
    st.session_state.read_loading = False

if st.session_state.read_output is not None:
    counts, address_qubits, data_qubits, cols, col_widths = st.session_state.read_output
    st.subheader("Read Result")
    plot_results_st(counts, address_qubits, data_qubits, cols, col_widths, bar_color='green')

st.button("Start Over", on_click=start_over, type="primary")