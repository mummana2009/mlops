FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory
WORKDIR /app

# Install required library for reading parquet
RUN pip install --no-cache-dir pandas pyarrow scikit-learn

# Copy your script into the image
COPY predict.py .

# Set the entrypoint
ENTRYPOINT ["python", "predict.py"]
