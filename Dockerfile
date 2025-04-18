# Step 1: Use a base image with Python
FROM python:3.10-slim

# Step 1.1: Update apt and install libgomp1
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Step 2: Set working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements.txt to the container
COPY requirements.txt .

# Step 4: Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the application files to the container
COPY . .

# Step 6: Expose the application port, we will use post 8000
EXPOSE 8000

# # Start the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]