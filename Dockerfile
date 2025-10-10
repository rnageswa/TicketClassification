# Stage 1: Build stage for installing dependencies
FROM python:3.10-slim AS builder

# Set the working directory
WORKDIR /app

# Install dependencies for deep learning libraries
# Needs to handle potentially large libraries like PyTorch and Hugging Face
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the application code
COPY ./app /app/app
COPY ./artifacts /app/artifacts
COPY ./src /app/src

# Set environment variables for non-interactive mode
ENV PYTHONUNBUFFERED=1

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
