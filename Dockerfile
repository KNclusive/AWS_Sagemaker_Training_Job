FROM python:3.11

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libusb-1.0-0-dev \
    libudev-dev \
    build-essential \
    ca-certificates && \
    rm -fr /var/lib/apt/lists/*

# Keep python from buffering the stdout - so the logs flushed quickly
ENV PYTHONUNBUFFERED=TRUE

# Don't compile bytecode
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV PATH="/opt/ml/code:${PATH}"
ENV PYTHONPATH=.

# Install packages
WORKDIR /opt/ml/code

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY code/ ./

# Add src code
RUN chmod +x train.py

# Set the entrypoint to your training script
ENTRYPOINT ["./train.py"]