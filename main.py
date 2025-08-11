from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from transformers import pipeline
import torch

# Initialize FastAPI app
app = FastAPI()

# Initialize the pipeline with the model (openai/gpt-oss-20b)
model_id = "openai/gpt-oss-20b"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,  # Explicitly use CPU-based precision
    device=-1,  # Set device=-1 to force CPU usage
)

# Define a static system prompt
system_prompt = "You are an AI assistant knowledgeable in various topics. Respond in a helpful, concise, and clear manner."


# Pydantic model for the input request
class LargeTextRequest(BaseModel):
    text: str  # Large text input (can be a large block of text)


# Function to chunk the large text into manageable pieces
def chunk_text(text, chunk_size=1024):
    """Chunk the input text into pieces smaller than the token limit."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for word in words:
        word_size = len(word.split())  # Simple word length approximation
        if current_chunk_size + word_size > chunk_size:
            # If the chunk is too large, start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_chunk_size = word_size
        else:
            # Otherwise, add the word to the current chunk
            current_chunk.append(word)
            current_chunk_size += word_size

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


@app.post("/generate")
async def generate_text(request: LargeTextRequest):
    try:
        # Retrieve the large text from the request
        large_text = request.text

        # Split the large text into chunks
        text_chunks = chunk_text(large_text, chunk_size=1024)

        # Store generated responses for each chunk
        generated_responses = []

        # Process each chunk of text
        for chunk in text_chunks:
            # Combine the system prompt and the chunk of text
            full_prompt = system_prompt + "\n" + chunk

            # Generate text based on the full prompt
            outputs = pipe([full_prompt], max_new_tokens=256)

            # Extract the generated text
            generated_text = outputs[0]["generated_text"]
            generated_responses.append(generated_text)

        # Combine all generated text chunks into one response
        final_output = " ".join(generated_responses)
        return {"generated_text": final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
