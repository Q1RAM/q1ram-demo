import streamlit as st
import platform
import os
import psutil
import datetime

st.set_page_config(
    page_title="Streamlit Server Info",
    page_icon="💻",
    layout="wide",
)

st.title("Server System Information (Streamlit Cloud)")
st.write("This app displays information about the server where it's hosted.")

# --- Basic Platform Information ---
st.header("1. Basic Platform Info")
st.code(f"""
System: {platform.system()}
Node Name: {platform.node()}
Release: {platform.release()}
Version: {platform.version()}
Machine: {platform.machine()}
Processor: {platform.processor()}
Python Version: {platform.python_version()}
""")

# --- CPU Information ---
st.header("2. CPU Information")
st.write(f"Physical cores: {psutil.cpu_count(logical=False)}")
st.write(f"Total cores (logical): {psutil.cpu_count(logical=True)}")

st.subheader("CPU Frequencies")
cpu_freq = psutil.cpu_freq()
if cpu_freq:
    st.write(f"Max Frequency: {cpu_freq.max:.2f} Mhz")
    st.write(f"Min Frequency: {cpu_freq.min:.2f} Mhz")
    st.write(f"Current Frequency: {cpu_freq.current:.2f} Mhz")
else:
    st.write("Could not retrieve CPU frequencies.")

st.subheader("CPU Usage")
# A short interval might be needed for accuracy on first call
cpu_percent = psutil.cpu_percent(interval=1)
st.write(f"Total CPU Usage: {cpu_percent}%")

st.subheader("CPU Usage Per Core:")
cpu_per_core = psutil.cpu_percent(percpu=True, interval=1)
for i, percentage in enumerate(cpu_per_core):
    st.write(f"Core {i}: {percentage}%")

# --- Memory Information ---
st.header("3. Memory Information")
svmem = psutil.virtual_memory()

def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
    return f"{bytes:.2f}{suffix}" # Fallback for extremely large numbers

st.write(f"Total: {get_size(svmem.total)}")
st.write(f"Available: {get_size(svmem.available)}")
st.write(f"Used: {get_size(svmem.used)}")
st.write(f"Percentage: {svmem.percent}%")

st.subheader("Swap Memory")
swap = psutil.swap_memory()
st.write(f"Total: {get_size(swap.total)}")
st.write(f"Free: {get_size(swap.free)}")
st.write(f"Used: {get_size(swap.used)}")
st.write(f"Percentage: {swap.percent}%")

# --- Disk Usage ---
st.header("4. Disk Usage")
# Get disk partitions
partitions = psutil.disk_partitions()
for partition in partitions:
    st.subheader(f"Device: {partition.device}")
    st.write(f"Mountpoint: {partition.mountpoint}")
    st.write(f"File system type: {partition.fstype}")
    try:
        partition_usage = psutil.disk_usage(partition.mountpoint)
        st.write(f"Total Size: {get_size(partition_usage.total)}")
        st.write(f"Used: {get_size(partition_usage.used)}")
        st.write(f"Free: {get_size(partition_usage.free)}")
        st.write(f"Percentage: {partition_usage.percent}%")
    except PermissionError:
        st.write("Permission denied to access this partition.")

# --- Network Information (Interfaces and IP Addresses) ---
st.header("5. Network Information")
if_addrs = psutil.net_if_addrs()
for interface_name, interface_addresses in if_addrs.items():
    st.subheader(f"Interface: {interface_name}")
    for address in interface_addresses:
        if str(address.family) == 'AddressFamily.AF_INET': # IPv4
            st.write(f"  IP Address: {address.address}")
            st.write(f"  Netmask: {address.netmask}")
            st.write(f"  Broadcast IP: {address.broadcast}")
        elif str(address.family) == 'AddressFamily.AF_PACKET': # MAC address
            st.write(f"  MAC Address: {address.address}")
        # Add more address families if needed (e.g., AF_INET6 for IPv6)

# --- Boot Time ---
st.header("6. Boot Time")
boot_time_timestamp = psutil.boot_time()
bt = datetime.datetime.fromtimestamp(boot_time_timestamp)
st.write(f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")

st.header("7. Environment Variables")
st.write("Some common environment variables:")
st.code(f"""
PATH: {os.environ.get('PATH', 'Not set')}
HOME: {os.environ.get('HOME', 'Not set')}
USER: {os.environ.get('USER', 'Not set')}
LANG: {os.environ.get('LANG', 'Not set')}
""")
if st.checkbox("Show all environment variables (caution: may contain sensitive info)"):
    st.json(dict(os.environ))

st.info("Note: Streamlit Cloud allocates resources dynamically. The reported values are for the container your app is running in, and may not reflect the entire underlying physical server.")