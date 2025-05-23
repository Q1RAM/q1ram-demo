import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

 
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
    column_max_bit_widths_list = [width for col, width in column_bit_widths.items()]


    return decimal_values, column_names, column_max_bit_widths_list

API_URL = "https://huggingface.co/spaces/ahmed-eisa/q1ram_DK"  # Change if deploying

st.title("QRAM API Streamlit Frontend")

# Step 1: Upload and preview Excel file
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    rows_values, cols, col_widths=process_excel_file(uploaded_file)
    rows_values = [int(x) for x in rows_values]
    col_widths = [int(x) for x in col_widths]
    df = pd.read_excel(uploaded_file)
    st.write("Preview of Excel data:")
    st.dataframe(df)

    # Step 2: Select columns to use as data
    read_addresses=st.multiselect("Select numeric columns to use as data", list(range(len(rows_values))))
    if read_addresses:
        try:
            # Step 3: Compute values for API
            # col_widths = [len(bin(int(df[c].max()))[2:]) for c in cols]
            # rows_values = df[cols].astype(int).apply(lambda row: int("".join(format(x, f"0{w}b") for x, w in zip(row, col_widths))), axis=1).tolist()
            # st.write("Encoded integer values:", rows_values)

            payload_base = {
                "rows_values": rows_values,
                "cols": cols,
                "col_widths": col_widths,
            }
            st.write("Payload for API:", payload_base)
            st.write("Column names:", cols)
            # Step 4: Buttons
            if st.button("Encode Data"):
                res = requests.post(f"{API_URL}/encode_data/", json=payload_base)
                st.image(Image.open(BytesIO(res.content)), caption="Encoded Data")

            if st.button("Write to QRAM"):
                res = requests.post(f"{API_URL}/write_qram/", json=payload_base)
                st.image(Image.open(BytesIO(res.content)), caption="QRAM Write")

            addresses = st.text_input("Optional: Read from specific address indices (comma-separated)", "")
            if st.button("Read from QRAM"):
                try:
                    address_list = read_addresses
                    payload = payload_base.copy()
                    payload["addresses"] = address_list
                    res = requests.post(f"{API_URL}/read_qram/", json=payload)
                    st.image(Image.open(BytesIO(res.content)), caption="QRAM Read")
                except ValueError:
                    st.error("Invalid address format. Please enter comma-separated integers.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
