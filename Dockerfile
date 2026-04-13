# Base runtime for the methane-processing CLI tools.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for rasterio, GDAL, opencv-python, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS custom
# Install Python dependencies
COPY requirements2.txt /tmp/requirements2.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements2.txt

FROM custom AS dev
WORKDIR /app

# Copy the runtime scripts used by the EOX workflow into the container.
COPY stac_search.py /app
COPY process_item.py /app
COPY aggregate_signals.py /app