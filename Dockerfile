# Use a base image with Python
FROM python:latest

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
CMD ["python", "app.py"]
