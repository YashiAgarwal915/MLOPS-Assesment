# 1. Start with a lean, official Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container to /app
WORKDIR /app

# 3. Copy only the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# 4. Install all the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project's code and files into the container
COPY . .

# 6. Specify the default command to run when the container starts
CMD ["python", "src/predict.py"]