import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

API_URL = "https://huggingface.co/spaces/ahmed-eisa/q1ram_DK"  # Change if deploying

st.title("QRAM API Streamlit Frontend")

# Step 1: Upload and preview Excel file
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Preview of Excel data:")
    st.dataframe(df)

    # Step 2: Select columns to use as data
    cols = st.multiselect("Select numeric columns to use as data", df.columns.tolist())
    if cols:
        try:
            # Step 3: Compute values for API
            col_widths = [len(bin(int(df[c].max()))[2:]) for c in cols]
            rows_values = df[cols].astype(int).apply(lambda row: int("".join(format(x, f"0{w}b") for x, w in zip(row, col_widths))), axis=1).tolist()
            st.write("Encoded integer values:", rows_values)

            payload_base = {
                "rows_values": rows_values,
                "cols": cols,
                "col_widths": col_widths,
            }

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
                    address_list = list(map(int, addresses.split(","))) if addresses.strip() else []
                    payload = payload_base.copy()
                    payload["addresses"] = address_list
                    res = requests.post(f"{API_URL}/read_qram/", json=payload)
                    st.image(Image.open(BytesIO(res.content)), caption="QRAM Read")
                except ValueError:
                    st.error("Invalid address format. Please enter comma-separated integers.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
