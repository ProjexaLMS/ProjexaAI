from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
import torch

app = FastAPI()

# Initialize the tokenizer and model
model_id = "unsloth/gpt-oss-20b-BF16"  # FP16-compatible model
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,  # Use FP16 precision
    device=-1,  # Use CPU
)

system_prompt = "You are an AI assistant knowledgeable in various topics. Respond in a helpful, concise, and clear manner."


class LargeTextRequest(BaseModel):
    text: str


def chunk_text(text, chunk_size=1024):
    """Chunk the input text into pieces smaller than the token limit."""
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= chunk_size:
            chunks.append(tokenizer.decode(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk))

    return chunks


@app.post("/generate")
async def generate_text(request: LargeTextRequest):
    try:
        large_text = request.text

        # Chunk the large text
        text_chunks = chunk_text(large_text, chunk_size=1024)

        generated_responses = []

        # Generate text for each chunk
        for chunk in text_chunks:
            full_prompt = system_prompt + "\n" + chunk
            outputs = pipe([full_prompt], max_new_tokens=256)
            generated_text = outputs[0]["generated_text"]
            generated_responses.append(generated_text)

        final_output = " ".join(generated_responses)
        return {"generated_text": final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
