# Install docker on remote machine and configure:
sudo apt update
sudo apt install -y docker.io
docker --version

# Now, start and enable Docker to run on boot:
sudo systemctl enable docker
sudo systemctl start docker

# Verify docker is running
sudo systemctl status docker

# Docker needs root access. To run Docker without sudo, add your user to the docker group:
sudo groupadd docker  # (only if the group doesn't exist)
sudo usermod -aG docker faysal.e73 (faysal.e73 is the username)

# Then apply the changes
newgrp docker

# Check if your user is in the docker group:
groups faysal.e73 (faysal.e73 is the username)




# This is our what project folder looks like:

        fashion-demand
                __pycache__
                .idea
                .env
                authentication.py
                Dockerfile
                main.py
                Official Training Data
                Official Training Data - Test
                requirements.txt
                subcat_forecast.py
                utilities.py

# Steps 1 to 5 is for local machine. We test the model in local machine and push the image to dockerhub
# Steps 5 to 12 is for remote machine
1. Create a docker file named "Dockerfile" and put the required contents below
        ## Step 1: Use a base image with Python
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

        # Start the FastAPI application using Uvicorn
        CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

2. Create a private repo named "fashion-demand" on dockerhub

3. Login to docker from the local machine. Enter docker hub username and passwword if asked
        docker login

4. Build the docker image using the command below
        docker build -t faysal37/fashion-demand:latest .

5. Push the docker image to dockerhub using
        docker push faysal37/fashion-demand:latest

# Steps 6 to 13 are for remote server
6. Login or SSH to the remote machine

7. Login to docker from remote server since the repo is private
        docker login

8. Pull the image from dockerhub to the remote machine
        docker pull faysal37/fashion-demand:latest

9. Install micro on remote to avoid using nano and verify the installation
        sudo apt install micro -y
        micro --version

10. Create a directory named "fashion-demand" inside home directory to store the .env

11. Create an env file named .env in fashion-demand folder, which will store the token required for authentication
        micro .env
        then add the line below and save
        SECRET_TOKEN=hashbin2

12. Start the docker container in detached mode from fashion-demand folder
        docker run -d --env-file .env -p 8000:8000 --name fashion faysal37/fashion-demand:latest

13. Update the restart policy from fashion-demand-folder
        docker update --restart=unless-stopped fashion





## Endpoints of the API:
1. http://34.172.217.151:8000/
        response:  {"status": "Healthy & Running"}
This endpoint just returns the satus of the server. 


2. http://34.172.217.151:8000/upload-train-data/
        response: {"message": "Train data uploaded successfully"}
This endpoint expects a csv format data. Without this data, the 3rd endpoind will not work


3. http://34.172.217.151:8000/make-forecast/
response:
        [
    "Total Demand for SKU: 162-2485 for 6 days (2025-02-22----2025-02-27): 2",
    "Reorder Alert: Reorder needed by Day 6.",
    "No overstock warning.",
    [
        {
            "sku": "162-2485",
            "date": "2024-12-30T00:00:00",
            "forecast": 0,
            "reorder_point": 31
        },

        {

        }
    ]]
We need to make sure we uploaded the data which is endpoint 2.

4. http://34.172.217.151:8000/download-forecast/
        response: a table of the forecast data from endpoint 3 that can be downloaded as csv.
        We need to make sure we make the forecast first before we send request to this endpoint.
