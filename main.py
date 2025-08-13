from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# ---- Choose a small CPU-friendly model ----
# Options that run on CPU:
#   - "EleutherAI/gpt-neo-125M"  (causal LM, tiny, fastest)
#   - "distilgpt2"               (also tiny, fast)
MODEL_ID = "EleutherAI/gpt-neo-125M"

# Load tokenizer & model for CPU
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32  # CPU-safe dtype
)

# Some tiny models have no pad token; map pad -> eos to silence warnings
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # force CPU
)

system_prompt = (
    "You are an AI assistant knowledgeable in various topics. "
    "Respond in a helpful, concise, and clear manner."
)


class LargeTextRequest(BaseModel):
    text: str


def chunk_text(text: str, chunk_size_tokens: int = 256):
    """
    Chunk input text into ~chunk_size_tokens pieces.
    Uses tokenizer to avoid splitting mid-token.
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0].tolist()

    chunks = []
    for i in range(0, len(ids), chunk_size_tokens):
        chunk_ids = ids[i : i + chunk_size_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks


@app.post("/generate")
async def generate_text(request: LargeTextRequest):
    try:
        large_text = request.text
        text_chunks = chunk_text(large_text, chunk_size_tokens=256)

        generated_responses = []
        for chunk in text_chunks:
            full_prompt = system_prompt + "\n" + chunk

            outputs = pipe(
                full_prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # text-generation pipeline returns a list of dicts
            generated_text = outputs[0]["generated_text"]
            generated_responses.append(generated_text)

        final_output = " ".join(generated_responses)
        return {"generated_text": final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
