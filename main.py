import os
import chromadb
import uuid
import yaml
import openai
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

db_path = config["database"]["path"]
chunk_size = config["ingestion"]["chunk_size"]
chunk_overlap = config["ingestion"]["chunk_overlap"]
max_prompt_length = config["llm"]["max_prompt_length"]
base_url = config["llm"]["base_url"]
model = config["llm"]["model"]
model_instructions = config["llm"]["instructions"]

if chunk_overlap >= chunk_size:
    raise ValueError("chunk_overlap must be less than chunk_size")

openai_api_key = os.environ.get("OPENAI_API_KEY")
api_key = os.environ.get("API_KEY")

if openai_api_key is None:
    raise OSError("No api key found at environment variable OPENAI_API_KEY.")

if api_key is None:
    raise OSError("No api key found at environment variable API_KEY.")

openai_client = openai.OpenAI(
    base_url=base_url,
    api_key = openai_api_key,
)

class Document(BaseModel):
    text: str

class IngestionResponse(BaseModel):
    document_id: str
    num_chunks: int
    status: str

class Query(BaseModel):
    prompt: str

class Response(BaseModel):
    content: str

app = FastAPI()

header_scheme = APIKeyHeader(name="api-key", auto_error=True)

client = chromadb.PersistentClient(path=db_path)

collection = client.get_or_create_collection("documents")

@app.post("/documents", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
def ingest(document: Document, key: str=Depends(header_scheme)):
    if key != api_key:
        raise HTTPException(403, "Invalid api key")
    text = document.text
    if text.strip() == "":
        raise HTTPException(422, "text is empty or whitespace")
    chunks = []
    for i in range(0, len(text), chunk_size-chunk_overlap):
        chunks.append(text[i:min(len(text), i+chunk_size)])
    document_id = str(uuid.uuid4())
    collection.add(
        ids=list([str(uuid.uuid4()) for _ in range(len(chunks))]),
        documents=chunks,
        metadatas=[{"document_id": document_id} for _ in range(len(chunks))]
    )
    return {
        "document_id": document_id,
        "num_chunks": len(chunks),
        "status": "ingested",
    }


@app.post("/query", response_model=Response, status_code=status.HTTP_200_OK)
def query(query: Query, key: str=Depends(header_scheme)):
    if key != api_key:
        raise HTTPException(403, "Invalid api key")
    prompt = query.prompt
    if prompt.strip() == "":
        raise HTTPException(422, "prompt is empty or whitespace")
    if len(prompt) > max_prompt_length:
        raise HTTPException(422, f"max prompt length exceeded ({max_prompt_length})")
    context_docs = collection.query(query_texts=[prompt])["documents"][0]
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": model_instructions},
                {"role": "user", "content": f"{prompt}\nRespond based on the following context:\n{"\n".join(context_docs)}"}
            ]
        )
    except openai.AuthenticationError:
        raise HTTPException(503, "Server has invalid api key for LLM")
    except openai.RateLimitError:
        raise HTTPException(429, "Rate Limit Exceeded")
    except:
        raise HTTPException(503, f"Unknown problem with openai or {base_url}")
    return {
        "content": response.choices[0].message.content,
    }