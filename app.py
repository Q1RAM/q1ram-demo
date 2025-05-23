# prompt: create a function the accepts an excel file , read it row by row,  concatenate the binary string of all column values in each row and convert it into decimal value, return array for all decimals, read  columns names ,skip the columns row from conversion, return also the column names and its max bit width in a tuble

# !pip install openpyxl

import pandas as pd
import numpy as np
from qiskit import *
from qiskit.circuit import Instruction


def normalize_to_statevector(classical_data):
  """
  Converts a list of tuples into a normalized state vector.

  Args:
    classical_data: A list of tuples, where each tuple (a, b) represents a
                    data point.

  Returns:
    np.array: A normalized state vector where the indices corresponding to
              concatenated decimal values (a concatenated with b) are 1/sqrt(N)
              and all other elements are 0. N is the number of data points.
  """
  n_max = 0
  m_max = 0
  for a, b in classical_data:
    n_max = max(n_max, a)
    m_max = max(m_max, b)

  n_bits = np.ceil(np.log2(len(classical_data))).astype(int)
  m_bits = np.ceil(np.log2(m_max + 1)).astype(int) # m_max + 1 because the max value b could be 0
  total_bits = n_bits + m_bits
  state_vector_size = 2**total_bits
  state_vector = np.zeros(state_vector_size, dtype=np.float32) # Use complex for state vectors

  num_data_points = len(classical_data)
  amplitude = 1.0 / np.sqrt(num_data_points)

  for i, (a, b) in enumerate(classical_data):
    # Pad 'a' and 'b' with leading zeros to match bit widths
    a_bin = bin(a)[2:].zfill(n_bits)
    b_bin = bin(b)[2:].zfill(m_bits)

    # Concatenate binary strings and convert back to decimal index
    concatenated_bin = a_bin + b_bin
    index = int(concatenated_bin, 2)

    # Assign amplitude to the corresponding index
    state_vector[index] = amplitude

  return state_vector

def indicies_to_statevector(indicies):
  n_max = max(indicies)
  


  n_bits = np.ceil(np.log2(len(classical_data))).astype(int)
  
  total_bits = n_bits 
  state_vector_size = 2**total_bits
  state_vector = np.zeros(state_vector_size, dtype=complex) # Use complex for state vectors

  num_data_points = len(indicies)
  amplitude = 1.0 / np.sqrt(num_data_points)

  for index in indicies:
    

    # Assign amplitude to the corresponding index
    state_vector[index] = amplitude

  return state_vector



def Simplified_RTOF_Gate(num_controls:int,num_targets=1,clean_ancilla=True)->Instruction:
    qc= QuantumCircuit()
    num_ancilla=num_controls//2-1
    controls=QuantumRegister(num_controls,name="control")
    ancilla=QuantumRegister(num_ancilla,name="ancilla")
    target=QuantumRegister(num_targets,name="target")
    # qc.add_register(controls)
    # qc.add_register(ancilla)
    # qc.add_register(target)

    if(num_controls==3):
      qc.add_register(controls)
      qc.add_register(target)
      qc.rcccx(controls[0],controls[1],controls[2],target)
    elif(num_controls==2):
      qc.add_register(controls)
      qc.add_register(target)
      qc.rccx(controls[0],controls[1],target)
    else:
      qc.add_register(controls)
      qc.add_register(ancilla)
      qc.add_register(target)
      if(num_ancilla!=num_controls//2-1):
          raise ValueError(f"Expected {num_controls//2-1} ancilla qubits, while {num_ancilla} is provided")

      i=0
      num_remaining_controls= num_controls
      while (num_remaining_controls>0):
          if(i==0):
              # print(f"RTOF: c{i*2},c{i*2+1},c{i*2+2}-->a{i}")
              qc.rcccx(controls[i*2],controls[i*2+1],controls[i*2+2],ancilla[i])
              num_remaining_controls-=3
          elif(num_remaining_controls>2):
              # print(f"RTOF: a{i-1},c{i*2+1},c{i*2+2}-->a{i}")
              qc.rcccx(ancilla[i-1],controls[i*2+1],controls[i*2+2],ancilla[i])
              num_remaining_controls-=2
          elif(num_remaining_controls==2):
              # print(f"RTOF: a{i-1},c{i*2+1},c{i*2+2}-->tar")
              for t in range(num_targets):
                qc.rcccx(ancilla[i-1],controls[i*2+1],controls[i*2+2],target[t])
              num_remaining_controls-=2
          elif(num_remaining_controls==1):
              # print(f"ccx:  a{i-1},c{num_controls-1}-->tar")
              for t in range(num_targets):
                qc.rccx(ancilla[i-1],controls[num_controls-1],target[t])
              num_remaining_controls-=1
          i+=1
      if(clean_ancilla):
          i=0
          num_remaining_controls= num_controls
          qc_temp=qc.copy_empty_like()
          while (num_remaining_controls>2):
            if(i==0):
                qc_temp.rcccx(controls[i*2],controls[i*2+1],controls[i*2+2],ancilla[i])
                num_remaining_controls-=3
            elif(num_remaining_controls>2):
                qc_temp.rcccx(ancilla[i-1],controls[i*2+1],controls[i*2+2],ancilla[i])
                num_remaining_controls-=2
            i+=1
          qc_temp=qc_temp.inverse()
          qc.compose(qc_temp,inplace=True)

    qc.name=f"({num_controls})RTOF-{'CLEAN' if clean_ancilla else ''}"
    return qc.to_instruction()

    
def process_excel_file(excel_filepath):
    """
    Reads an Excel file row by row, converts concatenated column binary strings
    to decimal, and returns the decimal values along with column names and their
    maximum bit widths.

    Args:
        excel_filepath (str): The path to the Excel file.

    Returns:
        tuple: A tuple containing:
            - list: A list of decimal values for each row.
            - list: A list of column names.
            - list: A list of tuples, where each tuple is (column_name, max_bit_width).
    """
    df = pd.read_excel(excel_filepath)

    decimal_values = []
    column_names = df.columns.tolist()
    column_bit_widths = {}

    # Determine the maximum bit width for each column
    for col in column_names:
        # Convert column data to string to handle potential non-numeric types
        col_data_str = df[col].astype(str)
        # Find the maximum length of the string representation of column values
        max_len = col_data_str.str.len().max()
        # Assuming each character represents a bit in the binary string
        column_bit_widths[col] = max_len if pd.notna(max_len) else 0


    for index, row in df.iterrows():
        binary_string = ""
        for col in column_names:
            # Convert each cell value to a string and append it to the binary string
            # Ensure that NaNs are treated as empty strings or handle as appropriate
            cell_value_str = str(row[col]) if pd.notna(row[col]) else ""
            binary_string += cell_value_str

        # Convert the concatenated binary string to decimal
        try:
            decimal_value = int(binary_string, 2)
            decimal_values.append(decimal_value)
        except ValueError:
            # Handle cases where the concatenated string is not a valid binary string
            print(f"Warning: Skipping row {index+2} due to invalid binary string: '{binary_string}'")
            decimal_values.append(None) # Or some other indicator for invalid data

    # Create the list of tuples for column names and max bit widths
    column_max_bit_widths_list = [(col, width) for col, width in column_bit_widths.items()]


    return decimal_values, column_names, column_max_bit_widths_list


from qiskit.circuit.library import UnitaryGate
def Classical_RAM_InterfaceGate(n,m,classical_data):
  qc_interface= QuantumCircuit()
  qr_AR=QuantumRegister(n,name="AR")
  qr_DR=QuantumRegister(m,name="DR")
  qr_Ar= QuantumRegister(n,name="Ar")
  qr_Cd= QuantumRegister(m,name="Cd")
  cb1=QuantumRegister(1,name="cb1")
  cb2=QuantumRegister(1,name="cb2")
  qr_tof_ancillae=QuantumRegister((n+m)//2-1,name="tof_ancillae")
  qc_interface.add_register(qr_AR,qr_DR,qr_Cd,qr_Ar,cb1,cb2,qr_tof_ancillae)


  qc_interface.x(cb2)
  for k in range(1,len(classical_data)+1):
    qc_interface.initialize(classical_data[k-1][0],qr_Ar)
    qc_interface.initialize(classical_data[k-1][1],qr_Cd)
    # Step 1
    for i in range(n):
      qc_interface.ccx(cb2,qr_Ar[i],qr_AR[i])

    for i in range(m):
      qc_interface.ccx(cb2,qr_Cd[i],qr_DR[i])


    # Step 2
    for i in range(n):
      qc_interface.cx(qr_Ar[i],qr_AR[i])
      qc_interface.x(qr_AR[i])

    for i in range(m):
      qc_interface.cx(qr_Cd[i],qr_DR[i])
      qc_interface.x(qr_DR[i])

    # Step 3
    R_ToffoliGate=Simplified_RTOF_Gate(n+m,clean_ancilla=True)
    qc_interface.append(R_ToffoliGate,[*qr_AR,*qr_DR]+[*qr_tof_ancillae]+[*cb1])
    # qc_interface.mcx([*qr_AR,*qr_DR],cb1)

    #Step 4
    if(k==1):
      M=len(classical_data)
    else:
      M=M-1

    CR_k_matrix=[[1,0,0,0],[0,1,0,0],[0,0,np.sqrt((M-1)/M),1/np.sqrt(M)],[0,0,-1/np.sqrt(M), np.sqrt((M-1)/M)]]
    CR_k=UnitaryGate(np.array(CR_k_matrix),label="CR_k")
    qc_interface.append(CR_k,[cb2,cb1])
    # qc_interface.ch(cb1,cb2)

    #Step 5
    qc_interface.append(R_ToffoliGate,[*qr_AR,*qr_DR]+[*qr_tof_ancillae]+[*cb1])
    # qc.mcx([*qr_AR,*qr_DR],cb1)
    # controls=[*qr_AR,*qr_DR]


    #Step 6
    for i in range(n):
      qc_interface.x(qr_AR[i])
      qc_interface.cx(qr_Ar[i],qr_AR[i])

    for i in range(m):
      qc_interface.x(qr_DR[i])
      qc_interface.cx(qr_Cd[i],qr_DR[i])

    #Step 7
    for i in range(n):
      qc_interface.ccx(cb2,qr_Ar[i],qr_AR[i])

    for i in range(m):
      qc_interface.ccx(cb2,qr_Cd[i],qr_DR[i])

    qc_interface.barrier()
  return qc_interface.to_instruction()



def encode_classical_data(qc,classical_data,qr_AR=None,qr_DR=None,qr_Ar=None,qr_Cd=None,cb1=None,cb2=None,qr_tof_ancillae=None):
  if(qr_AR):
    n=len(qr_AR)
  else:
    n=np.ceil(np.log2(len(classical_data))).astype(int)
  if(qr_DR):
    m=len(qr_DR)
  else:
    m=np.ceil(np.log2(max(classical_data,key=lambda x:x[1])[1])).astype(int)
    
  if(qr_AR is None):
    qr_AR=QuantumRegister(n,name="AR")
    qc.add_register(qr_AR)
  if(qr_DR is None ):
    qr_DR=QuantumRegister(m,name="DR")
    qc.add_register(qr_DR)
  if(qr_Ar is None):
    qr_Ar= QuantumRegister(n,name="Ar")
    qc.add_register(qr_Ar)
  if(qr_Cd is None):
    qr_Cd= QuantumRegister(m,name="Cd")
    qc.add_register(qr_Cd)
  if(cb1 is None):
    cb1=QuantumRegister(1,name="cb1")
    qc.add_register(cb1)
  if(cb2 is None):
    cb2=QuantumRegister(1,name="cb2")
    qc.add_register(cb2)

  if(qr_tof_ancillae is None):
    qr_tof_ancillae=QuantumRegister((n+m)//2-1,name="tof_ancillae")
    qc.add_register(qr_tof_ancillae)
  qc.append(Classical_RAM_InterfaceGate(n,m,classical_data),[*qr_AR,*qr_DR,*qr_Cd,*qr_Ar,*cb1,*cb2,*qr_tof_ancillae])
  return qr_AR,qr_DR,qr_Ar,qr_Cd,cb1,cb2,qr_tof_ancillae


import matplotlib.pyplot as plt
def add_hyphens_to_bitstring(bitstring, col_widths):
  """
  Adds hyphens between the bitstring segments corresponding to different columns.

  Args:
      bitstring (str): The concatenated binary string.
      col_widths (list): A list of tuples, where each tuple is
                         (column_name, bit_width) representing the bit width
                         of each column in the original concatenated string.

  Returns:
      str: The bitstring with hyphens inserted.
  """
  formatted_bitstring = ""
  current_index = 0
  for col_name, width in col_widths:
    if width > 0:
      formatted_bitstring += bitstring[current_index : current_index + width]
      formatted_bitstring += "-"
      current_index += width
  # Remove the trailing hyphen
  return formatted_bitstring.rstrip('-')


def plot_results(counts,fig_name,n,m,cols,col_widths):
  address_qubits=n
  data_qubits=m
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
      # x_labels.append(f"{address[-address_qubits:]} ({address[:data_qubits]})")
      x_labels.append(f"{address[-address_qubits:]}:({formatted_address_data})")

  # Create the plot
  plt.figure(figsize=(12, 6))
  bars = plt.bar(x_labels, counts_values)

  # Color bars with zero counts light gray
  for i, bar in enumerate(bars):
      if counts_values[i] == 0:
          bar.set_color('lightgray')

  # Identify and color repeated addresses
  address_counts = {}
  for address in address_values:
    address_counts[address[-address_qubits:]] = address_counts.get(address[-address_qubits:], 0) + 1
  for i, bar in enumerate(bars):
    if address_counts[address_values[i][-address_qubits:]] > 1 and int(address_values[i][:data_qubits])==0:
      bar.set_color('lightgray')

  # Add legend
  light_gray_patch = plt.Rectangle((0, 0), 1, 1, fc="lightgray")
  plt.legend([light_gray_patch], ["Empty cell value"])

  plt.xlabel(f"Address:({'- '.join(cols)})")
  plt.ylabel("Counts")
  plt.title("QRAM Read Results")
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.savefig(f"{fig_name}")



from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator,StatevectorSimulator
from qiskit.visualization import plot_histogram
def simulate_circuit(qc):
  simulator = AerSimulator(method="matrix_product_state")
  compiled_circuit = transpile(qc.decompose().decompose().decompose(), simulator,optimization_level=3)
  job = simulator.run(compiled_circuit, shots=1024)
  result = job.result()
  counts = result.get_counts()
  return counts
