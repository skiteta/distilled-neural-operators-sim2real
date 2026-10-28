FROM python:3.10.16-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.2.2+cu121 numpy==1.26.4 h5py==3.11.0 PyYAML==6.0.2 filelock==3.29.0 typing_extensions==4.16.0 sympy==1.14.0 mpmath==1.3.0 networkx==3.4.2 Jinja2==3.1.6 MarkupSafe==3.0.3 fsspec==2026.4.0 nvidia-nvjitlink-cu12==12.1.105
WORKDIR /workspace
COPY . /workspace
