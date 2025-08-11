# Step 1: Start with a Python image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the application files into the container
COPY ./main.py /app/app.py

# Step 4: Install the necessary Python dependencies
RUN pip install --no-cache-dir \
    torch==1.13.0+cpu \
    transformers==4.25.0 \
    fastapi==0.95.0 \
    uvicorn==0.20.0

# Step 5: Expose the port that FastAPI will run on
EXPOSE 8000

# Step 6: Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
