import streamlit as st
import pandas as pd
import os
import glob
import numpy as np # Added for np.ceil, np.log2, np.max, np.astype
from Q1RAM import *
from ClassicalQuantumGateway import *

# --- Streamlit Application ---

st.set_page_config(layout="wide") # Use wide layout for better space utilization

# Initialize session state for persistence
if 'data_properties' not in st.session_state:
    st.session_state.data_properties = {
        "excel_data": None,
        "rows_values": [],
        "cols": [],
        "col_widths": []
    }
if 'show_step2' not in st.session_state:
    st.session_state.show_step2 = False
if 'show_step3' not in st.session_state:
    st.session_state.show_step3 = False
if 'show_step4' not in st.session_state:
    st.session_state.show_step4 = False
if 'encoding_image' not in st.session_state:
    st.session_state.encoding_image = None
if 'write_image' not in st.session_state:
    st.session_state.write_image = None
if 'read_image' not in st.session_state:
    st.session_state.read_image = None
if 'address_choices' not in st.session_state:
    st.session_state.address_choices = []


def remove_results():
    directory = './'
    pattern = os.path.join(directory, '*result*.png')
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
            # print(f"Deleted: {file_path}") # Don't print in Streamlit on every rerun
        except Exception as e:
            st.warning(f"Error deleting {file_path}: {e}") # Use st.warning for Streamlit


# Step 1: Load Excel file
st.header("### Step 1: Upload Excel File")
upload_btn = st.file_uploader("Upload dataset .xlsx", type=['xlsx'])

if upload_btn is not None:
    # Save uploaded file temporarily to process it with pandas
    file_path = os.path.join("./temp_upload", upload_btn.name)
    os.makedirs("./temp_upload", exist_ok=True) # Ensure dir exists
    with open(file_path, "wb") as f:
        f.write(upload_btn.getbuffer())

    excel_data = pd.read_excel(file_path)
    row_count = len(excel_data)
    rows_values, cols, col_widths = process_excel_file(file_path)
    remove_results() # Clear previous results on new upload

    st.session_state.data_properties["excel_data"] = excel_data
    st.session_state.data_properties["rows_values"] = rows_values
    st.session_state.data_properties["cols"] = cols
    st.session_state.data_properties["col_widths"] = col_widths
    st.session_state.address_choices = list(range(row_count))
    
    st.session_state.show_step2 = True # Show next steps
    st.session_state.show_step3 = False # Hide subsequent steps on new upload
    st.session_state.show_step4 = False
    
    st.subheader("Loaded Excel Data")
    st.dataframe(excel_data, use_container_width=True)
    st.code("classical_data = pd.read_excel(...)", language="python")


# Step 2: Classical to Quantum Encoding
if st.session_state.show_step2:
    st.header("### Step 2: Classical to Quantum Encoding")
    col1_s2, col2_s2 = st.columns([1, 2])
    with col1_s2:
        st.code("encode_classical_data_in_superposition(excel_file)", language="python")
        if st.button("Execute Encoding"):
            # Call your encoding function
            excel_data = st.session_state.data_properties["excel_data"]
            rows_values = st.session_state.data_properties["rows_values"]
            cols = st.session_state.data_properties["cols"]
            col_widths = st.session_state.data_properties["col_widths"]

            address_qubits = np.ceil(np.log2(len(rows_values))).astype(int)
            data_qubits = np.ceil(np.log2(np.max(rows_values))).astype(int)
            classical_data = list(zip(range(len(rows_values)), rows_values))
            qc = QuantumCircuit()
            qr_AR, qr_DR, qr_Ar, qr_Cd, cb1, cb2, qr_tof_ancillae = encode_classical_data(qc, classical_data)
            
            cr_address = ClassicalRegister(address_qubits, name='cr_address') # Added name
            cr_data = ClassicalRegister(data_qubits, name='cr_data') # Added name
            cr_read_flag = ClassicalRegister(1, name='cr_read_flag') # Added name
            qc.add_register(cr_address, cr_data, cr_read_flag)
            qc.measure(qr_AR, cr_address)
            qc.measure(qr_DR, cr_data)
            
            counts = simulate_circuit(qc)
            plot_results(counts, "encode_data_result.png", address_qubits, data_qubits, cols, col_widths)
            st.session_state.encoding_image = "encode_data_result.png"
            st.session_state.show_step3 = True # Show next step
            st.session_state.show_step4 = False # Hide subsequent steps
            st.rerun() # Rerun to update visibility

    with col2_s2:
        if st.session_state.encoding_image:
            st.image(st.session_state.encoding_image, caption="Encoding Result")

# Step 3: Write into QRAM
if st.session_state.show_step3:
    st.header("### Step 3: Write into QRAM")
    col1_s3, col2_s3 = st.columns([1, 2])
    with col1_s3:
        st.code("qram.write()", language="python")
        if st.button("Execute Writing"):
            excel_data = st.session_state.data_properties["excel_data"]
            rows_values = st.session_state.data_properties["rows_values"]
            cols = st.session_state.data_properties["cols"]
            col_widths = st.session_state.data_properties["col_widths"]

            address_qubits = np.ceil(np.log2(len(rows_values))).astype(int)
            data_qubits = np.ceil(np.log2(np.max(rows_values))).astype(int)
            classical_data = list(zip(range(len(rows_values)), rows_values))
            qc = QuantumCircuit()
            qr_AR, qr_DR, qr_Ar, qr_Cd, cb1, cb2, qr_tof_ancillae = encode_classical_data(qc, classical_data)
            qram = Q1RAM(address_qubits, data_qubits, qc=qc, qr_address_bus=qr_AR, qr_data_bus=qr_DR)
            qram.apply_write()
            qram.Measure_Internal_Data()

            counts = simulate_circuit(qc)
            plot_results(counts, "write_result.png", address_qubits, data_qubits, cols, col_widths)
            st.session_state.write_image = "write_result.png"
            st.session_state.show_step4 = True # Show next step
            st.rerun() # Rerun to update visibility

    with col2_s3:
        if st.session_state.write_image:
            st.image(st.session_state.write_image, caption="Write Result")

# Step 4: Read Arbitrary Address(es)
if st.session_state.show_step4:
    st.header("### Step 4: Read Arbitrary Address(es)")
    col1_s4, col2_s4 = st.columns([1, 2])
    with col1_s4:
        # CheckboxGroup in Streamlit requires initial options
        selected_addresses = st.multiselect(
            label="Select Address(es)",
            options=st.session_state.address_choices,
            default=[] # No addresses selected by default
        )
        st.code("qram.read()", language="python")
        if st.button("Execute Reading"):
            excel_data = st.session_state.data_properties["excel_data"]
            rows_values = st.session_state.data_properties["rows_values"]
            cols = st.session_state.data_properties["cols"]
            col_widths = st.session_state.data_properties["col_widths"]

            address_qubits = np.ceil(np.log2(len(rows_values))).astype(int)
            data_qubits = np.ceil(np.log2(np.max(rows_values))).astype(int)
            classical_data = list(zip(range(len(rows_values)), rows_values))
            qc = QuantumCircuit()
            qr_AR, qr_DR, qr_Ar, qr_Cd, cb1, cb2, qr_tof_ancillae = encode_classical_data(qc, classical_data)
            qram = Q1RAM(address_qubits, data_qubits, qc=qc, qr_address_bus=qr_AR, qr_data_bus=qr_DR)
            qram.apply_write()
            
            # Resetting registers as in original code
            # Note: For mock QuantumCircuit, these resets might not do much visually
            # but they keep the logic consistent with original.
            qc.reset(qr_AR)
            qc.reset(qr_DR)
            qc.reset(qr_Ar)
            qc.reset(qr_Cd)
            qc.reset(cb1)
            qc.reset(cb2)
            qc.reset(qr_tof_ancillae)
            
            if len(selected_addresses) > 0:
                qram.qc.prepare_state(indicies_to_statevector(selected_addresses), qr_AR)
                qram.apply_read()
            else:
                qram.ReadAll()

            qram.Measure()

            counts = simulate_circuit(qc)
            plot_results(counts, "read_result.png", address_qubits, data_qubits, cols, col_widths)
            st.session_state.read_image = "read_result.png"
            st.rerun() # Rerun to update image

    with col2_s4:
        if st.session_state.read_image:
            st.image(st.session_state.read_image, caption="Read Result")

    # Start Over button
    if st.button("Start Over"):
        remove_results()
        st.session_state.show_step2 = False
        st.session_state.show_step3 = False
        st.session_state.show_step4 = False
        st.session_state.data_properties = { # Reset data properties as well
            "excel_data": None,
            "rows_values": [],
            "cols": [],
            "col_widths": []
        }
        st.session_state.encoding_image = None
        st.session_state.write_image = None
        st.session_state.read_image = None
        st.session_state.address_choices = []
        st.rerun() # Rerun the entire app to reset