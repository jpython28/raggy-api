import os
import chromadb
import uuid
import yaml
import openai
import logging
import time
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(asctime)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

db_path = config["database"]["path"]
chunk_size = config["ingestion"]["chunk_size"]
chunk_overlap = config["ingestion"]["chunk_overlap"]
max_ingestion_characters = config["ingestion"]["max_ingestion_characters"]
max_prompt_length = config["llm"]["max_prompt_length"]
base_url = config["llm"]["base_url"]
model = config["llm"]["model"]
model_instructions = config["llm"]["instructions"]
similarity_threshold = config["llm"]["similarity_threshold"]
openai_timeout = config["llm"]["openai_timeout"]

logger.debug("Config loaded", extra=config)

if chunk_overlap >= chunk_size:
    raise ValueError("chunk_overlap must be less than chunk_size")

openai_api_key = os.environ.get("OPENAI_API_KEY")
api_key = os.environ.get("API_KEY")

if openai_api_key is None:
    logger.critical("environment variable OPENAI_API_KEY missing")
    raise OSError("No api key found at environment variable OPENAI_API_KEY.")

if api_key is None:
    logger.critical("environment variable API_KEY missing")
    raise OSError("No api key found at environment variable API_KEY.")

openai_client = openai.OpenAI(
    base_url=base_url,
    api_key = openai_api_key,
    timeout=openai_timeout,
)

class Document(BaseModel):
    text: str

class IngestionResponse(BaseModel):
    document_id: str
    num_chunks: int
    status: str

class Query(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    content: str
    chunks_used: int

class HealthSummary(BaseModel):
    server_status: str
    chroma_status: str

app = FastAPI()

header_scheme = APIKeyHeader(name="api-key", auto_error=False)

chroma_client = chromadb.PersistentClient(path=db_path)

collection = chroma_client.get_or_create_collection("documents")

logger.info("Server started")

@app.post("/documents", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
def ingest(document: Document, key: str=Depends(header_scheme)):
    start_time = time.perf_counter()
    if key != api_key:
        logger.warning("Invalid API key received")
        raise HTTPException(401, "Invalid or missing api key")
    text = document.text
    if len(text) > max_ingestion_characters:
        raise HTTPException(422, f"Document exceeds character limit ({max_ingestion_characters})")
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
    logger.info("Document ingested", extra={"document_id": document_id, "num_chunks": len(chunks), "latency": time.perf_counter()-start_time})
    return {
        "document_id": document_id,
        "num_chunks": len(chunks),
        "status": "ingested",
    }


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query(query: Query, key: str=Depends(header_scheme)):
    start_time = time.perf_counter()
    if key != api_key:
        logger.warning("Invalid or missing API key received")
        raise HTTPException(401, "Invalid or missing api key")
    prompt = query.prompt
    if prompt.strip() == "":
        raise HTTPException(422, "prompt is empty or whitespace")
    if len(prompt) > max_prompt_length:
        raise HTTPException(422, f"max prompt length exceeded ({max_prompt_length})")
    logging.debug("Querying chroma database", extra={"prompt": prompt})
    query_result = collection.query(query_texts=[prompt])
    chunks = query_result["documents"][0]
    distances = query_result["distances"][0]
    context_chunks = list([chunks[i] for i in range(len(chunks)) if distances[i] <= similarity_threshold])
    if len(context_chunks) > 0:
        context_prompt = f"{prompt}\nRespond based on the following context:\n{"\n".join(context_chunks)}"
    else:
        context_prompt = f"{prompt}\nNo relevant context was found."
        logger.warning("No chunks cleared similarity threshold")
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": model_instructions},
                {"role": "user", "content": context_prompt}
            ]
        )
    except openai.AuthenticationError:
        raise HTTPException(503, "Server has invalid api key for LLM")
    except openai.RateLimitError:
        logger.warning("openai rate limit exceeded")
        raise HTTPException(429, "Rate limit exceeded")
    except openai.APITimeoutError:
        logger.warning("openai timeout exceeded")
        raise HTTPException(504, f"openai timeout exceeded ({openai_timeout} secs)")
    except:
        logger.error("openai call failed for unknown reason")
        raise HTTPException(503, f"Unknown problem with openai or {base_url}")
    logger.info("Query handled", extra={"chunks_used": len(context_chunks), "latency": time.perf_counter()-start_time})
    return {
        "content": response.choices[0].message.content,
        "chunks_used": len(context_chunks),
    }

@app.get("/health", response_model=HealthSummary, status_code=status.HTTP_200_OK)
def get_health():
    chroma_status = "ok"
    try:
        chroma_client.heartbeat()
    except:
        logger.critical("Chroma database unreachable")
        chroma_status = "unreachable"
    logger.info("Health checked")
    return {
        "server_status": "ok",
        "chroma_status": chroma_status
    }