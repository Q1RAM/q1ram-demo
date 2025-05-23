import streamlit as st
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import matplotlib as qiskit_matplotlib
import matplotlib.pyplot as plt

def plot_results_st(counts, n, m, cols, col_widths, bar_color='blue'):
    address_qubits = n
    data_qubits = m
    sorted_counts = dict(sorted(counts.items(), key=lambda item: int(item[0][-address_qubits:], 2)))

    # Prepare data for plotting
    address_values = []
    counts_values = []
    for address, count in sorted_counts.items():
        address_values.append(address)
        counts_values.append(count)

    # Modify x labels
    x_labels = []
    for address in address_values:
        formatted_address_data = add_hyphens_to_bitstring(address[:data_qubits], col_widths)
        x_labels.append(f"{address[-address_qubits:]}:({formatted_address_data})")

    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x_labels, counts_values, color=bar_color)

    # Color bars with zero counts light gray
    for i, bar in enumerate(bars):
        if counts_values[i] == 0:
            bar.set_color('lightgray')

    # Identify and color repeated addresses
    address_counts = {}
    for address in address_values:
        address_counts[address[-address_qubits:]] = address_counts.get(address[-address_qubits:], 0) + 1
    for i, bar in enumerate(bars):
        if address_counts[address_values[i][-address_qubits:]] > 1 and int(address_values[i][:data_qubits]) == 0:
            bar.set_color('lightgray')

    # Add legend
    light_gray_patch = plt.Rectangle((0, 0), 1, 1, fc="lightgray")
    plt.legend([light_gray_patch], ["Empty cell value"])

    plt.xlabel(f"Address:({'- '.join(cols)})")
    plt.ylabel("Counts")
    plt.title("QRAM Read Results")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Display plot in Streamlit
    st.pyplot(plt.gcf())
    plt.close()

excel_data=[]
rows_values=[]
cols=[]
col_widths=[]

# Step 1: Load Excel file and return number of rows
def load_excel(file):
    global excel_data
    global rows_values
    global cols
    global col_widths


    excel_data = pd.read_excel(file.name)
    row_count = len(excel_data)
    rows_values, cols, col_widths = process_excel_file(file.name)
    

# Step 2: Encode classical data to superposition 
def encode_data():
    global excel_data
    global rows_values
    global cols
    global col_widths

    address_qubits= np.ceil(np.log2(len(rows_values))).astype(int)
    data_qubits=np.ceil(np.log2(np.max(rows_values))).astype(int)
    classical_data=list(zip(range(len(rows_values)),rows_values))
    qc=QuantumCircuit()
    qr_AR,qr_DR,qr_Ar,qr_Cd,cb1,cb2,qr_tof_ancillae=encode_classical_data(qc,classical_data)
    cr_address=ClassicalRegister(address_qubits)
    cr_data=ClassicalRegister(data_qubits)
    cr_read_flag=ClassicalRegister(1)
    qc.add_register(cr_address,cr_data)
    qc.measure(qr_AR,cr_address)
    qc.measure(qr_DR,cr_data)
    counts=simulate_circuit(qc)
    plot_results_st(counts,address_qubits,data_qubits,cols,col_widths,bar_color='red')
    

# Step 3: Write into QRAM 
def write_qram():
    global excel_data
    global rows_values
    global cols
    global col_widths

    address_qubits= np.ceil(np.log2(len(rows_values))).astype(int)
    data_qubits=np.ceil(np.log2(np.max(rows_values))).astype(int)
    classical_data=list(zip(range(len(rows_values)),rows_values))
    qc=QuantumCircuit()
    qr_AR,qr_DR,qr_Ar,qr_Cd,cb1,cb2,qr_tof_ancillae=encode_classical_data(qc,classical_data)
    qram=Q1RAM(address_qubits,data_qubits,qc=qc,qr_address_bus=qr_AR,qr_data_bus=qr_DR)
    qram.apply_write()
    qram.Measure_Internal_Data()

    counts=simulate_circuit(qc)
    plot_results_st(counts,address_qubits,data_qubits,cols,col_widths,bar_color='green')
    

# Step 4: Read from QRAM 
def read_qram(addresses):
    global excel_data
    global rows_values
    global cols
    global col_widths

    address_qubits= np.ceil(np.log2(len(rows_values))).astype(int)
    data_qubits=np.ceil(np.log2(np.max(rows_values))).astype(int)
    classical_data=list(zip(range(len(rows_values)),rows_values))
    qc=QuantumCircuit()
    qr_AR,qr_DR,qr_Ar,qr_Cd,cb1,cb2,qr_tof_ancillae=encode_classical_data(qc,classical_data)
    qram=Q1RAM(address_qubits,data_qubits,qc=qc,qr_address_bus=qr_AR,qr_data_bus=qr_DR)
    qram.apply_write()
    qram.qc.reset(qr_AR)
    qram.qc.reset(qr_DR)
    qram.qc.reset(qr_Ar)
    qram.qc.reset(qr_Cd)
    qram.qc.reset(cb1)
    qram.qc.reset(cb2)
    qram.qc.reset(qr_tof_ancillae)
    
    if(len(addresses)>0):
      qram.qc.prepare_state(indicies_to_statevector(addresses),qr_AR)
    # qram.ReadAll()
      qram.apply_read()
    else:
      qram.ReadAll()

    qram.Measure()

    counts=simulate_circuit(qc)
    plot_results_st(counts,address_qubits,data_qubits,cols,col_widths,bar_color='green')
    