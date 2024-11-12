# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter Lab
RUN pip install jupyterlab

# Copy the rest of the app code
COPY . .

# Expose ports for Streamlit and Jupyter Lab
EXPOSE 8501 8888 5000