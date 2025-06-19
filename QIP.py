from qiskit import *
from PIL import Image
import numpy as np

from qiskit.circuit.library import UCRYGate




def load_image(source_image,new_size=None):
  img_array=source_image
  if(isinstance( source_image , str)):
      img = Image.open(source_image).convert('L')
      if(new_size):
        img= img.resize(new_size)
      img_array = np.array(img)
  return img_array



def FRQI_encoding(qc,source_image,qr_position=None,qr_color=None,apply_h=True):
  n = int(np.ceil(np.log2(max(source_image.shape[:2]))))  # Number of qubits for positions
  if(qr_position is None):
    qr_position= QuantumRegister(2*n,name="Position")
    qc.add_register(qr_position)
  if(qr_color is None):
    qr_color= QuantumRegister(1,name="Color")
    qc.add_register(qr_color)

  qc_copy=qc.copy_empty_like(name=f"FRQI Image Encoding")
  if(apply_h):
    qc_copy.h(qr_position)

  flat_image = source_image.flatten()
  angles_list = list(2.0*np.arcsin(flat_image / 255.0))
  # print(source_image.shape)
  ucry= UCRYGate(angle_list=angles_list)
  qubits=[*qr_position]+[*qr_color]
  qc_copy.append(ucry,reversed(qubits))
  OFRQI_Gate= qc_copy.to_instruction()
  qc.append(OFRQI_Gate,qc.qubits)
  # qc.barrier()
  return qr_position,qr_color



def decode_frqi_image_aer(probabilities,n,use_zero_state=False):
  if(use_zero_state):
    filtered_dict = {k.replace(" ",""): v for k, v in probabilities.items() if k.startswith("0")}
  else:
    filtered_dict = {k.replace(" ",""): v for k, v in probabilities.items() if k.startswith("1")}

  restored_image= np.zeros((2**n,2**n))
  temp={}
  for k,v in filtered_dict.items():
    if(k in temp):
      print(f"repeated value:- key:{k} ,old:{temp[k]},new{v}")
    prob=0
    if(k not in temp or temp[k]<v):
      if(use_zero_state):
        prob= v-(1/(2**(2*n)))
      else:
        prob= v #if v>=0 else 0
      temp[k]=v
    xy_b=k[1:]
    x=xy_b[0:n][::-1]
    y=xy_b[n:2*n][::-1]
    val= int(255.0*np.sqrt(prob)*float(2**(2*n)))
    # print(v)
    # print(f"key:{k},x:{x},y:{y}")
    restored_image[int(y,2),int(x,2)]=val
  return restored_image

def group_dict_by_prefix(input_dict: dict, k: int) -> dict:
    """
    Splits a dictionary of bitstring-probability pairs into a larger dictionary
    where keys are prefixes and values are sub-dictionaries of matching entries.

    Args:
        input_dict (dict): The original dictionary where keys are bitstrings
                           and values are probabilities.
        k (int): The length of the bitstring prefix to group by.

    Returns:
        dict: A dictionary where:
              - Keys are the 'k'-bit prefixes.
              - Values are dictionaries containing the original bitstrings
                and their probabilities that start with that prefix.
    """
    grouped_data = {}
    for bitstring, probability in input_dict.items():
        if len(bitstring) < k:
            # Handle cases where a bitstring is shorter than the desired prefix length
            # You might want to raise an error, skip, or pad the bitstring
            print(f"Warning: Bitstring '{bitstring}' (length {len(bitstring)}) is shorter than prefix length {k}. Skipping.")
            continue

        prefix = bitstring[:k]
        suffix= bitstring[k:]
        # If the prefix is not yet a key in grouped_data, initialize it with an empty dictionary
        if prefix not in grouped_data:
            grouped_data[prefix] = {}

        # Add the original bitstring and its probability to the sub-dictionary
        grouped_data[prefix][suffix] = probability

    return grouped_data

def implement_boolean_function_ucry(qc:QuantumCircuit,qr_controls,qr_target,truth_table):
    non_zero_count = np.count_nonzero(truth_table)
    # zero_count=len(truth_table)-non_zero_count
    # if(zero_count<0.5*len(truth_table)):
    #   qc.x(qr_target)
    #   angles_list = np.pi*np.array(1-truth_table)
    # else:
    angles_list = np.pi*np.array(truth_table)

    ucry= UCRYGate(angle_list=list(angles_list))
    qubits=list(reversed([*qr_controls]))+[qr_target]
    # qc.h(qr_controls)

    qc.append(ucry,reversed(qubits))


  
def image_to_color_truth_table(image_array, color_bits=8, rescale=False):
    """
    Convert a grayscale image array to a truth table of color bits for each pixel.

    Args:
        image_array (numpy.ndarray): The grayscale image array with shape (height, width).
        color_bits (int): The number of bits to represent the color value.
        rescale (bool): Whether to rescale color values to the full color range.

    Returns:
        numpy.ndarray: A 2D array where each row represents a pixel and each column a specific color bit.
    """

    # Flatten the image into a 1D array
    pixel_data = image_array.flatten()

    # Rescale color values if requested
    if rescale:
        pixel_data = (pixel_data.astype(np.float32) * (2**color_bits - 1) / 255.0).astype(np.uint8)
    else:
        pixel_data = pixel_data.astype(np.uint8)

    # Create a truth table for each bit in the color value
    truth_table = np.array([(pixel_data >> bit) & 1 for bit in range(color_bits)])

    return truth_table

def encode_color_in_position_with_UCRY(image,qc,position_register=None,color_register=None,img_index="",apply_h=True,q=8):
    # Get the dimensions of the image
    height, width = image.shape
    m = int(np.log2(height))  # Assume height == width and is a power of 2
    n = int(np.log2(width))

    if(position_register is None):
        position_register = QuantumRegister(m+n, name=f'pos_{img_index}')
        qc.add_register(position_register)
    if(color_register is None):
        color_register = QuantumRegister(q, name=f'color_{img_index}')
        qc.add_register(color_register)

    qc_copy= qc.copy_empty_like()
    qc_copy.name=f"UCRY NEQR Image {height}x{width}_POS"

    if(apply_h):
      qc_copy.h(position_register)

    color_functions= image_to_color_truth_table(image,color_bits=q)

    if(color_register):
      for i, color_func in enumerate(color_functions):
          implement_boolean_function_ucry(qc_copy,position_register,color_register[i],color_func)

    PrepareGate= qc_copy.to_instruction()

    qc.append(PrepareGate,range(qc.num_qubits))
    return position_register,color_register


def reconstruct_neqr_image(probabilities,rows_qubits, col_qubits,color_qubits, filter_probs=False):

    if(filter_probs):
      filtered_dict = {k.replace(" ",""):v for k,v in probabilities.items() }#if   v>0.0001 }
    else:
      filtered_dict= probabilities


    num_rows= 2**rows_qubits
    num_cols= 2**col_qubits
    total_qubits=rows_qubits+col_qubits
    # Create an empty image array
    img_array = np.zeros((num_rows, num_cols), dtype=np.uint8)

    # Iterate over the dictionary
    for key, value in filtered_dict.items():
        # Convert binary row and column indices to decimal
        val=key[:color_qubits]
        pos=key[color_qubits:]
        col_index = int(pos[col_qubits:], 2)
        row_index = int(pos[:rows_qubits], 2)
        # Set the pixel value in the image array
        # print(f"key:{key},y:{row_index},x:{col_index},color-->{val}")
        if(int(val,2)>0):
          img_array[row_index, col_index] =int(val,2)

    # display(img_array)
    # Convert to uint8
    img_array = img_array.astype(np.uint8)


    # Create grayscale image from array
    img = Image.fromarray(img_array)

    return img