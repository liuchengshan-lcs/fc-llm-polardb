from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
from schema import EmbeddingRequest, EmbeddingResponse, SimilarityResponse
import service
app = FastAPI()


@app.get("/")
def index():
    return RedirectResponse(url="/docs")


@app.post("/embedding", response_model=EmbeddingResponse)
async def encode(request: EmbeddingRequest):
    return {
        "object": "list",
        "data": service.encode(request.sentences)
    }


@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(request: EmbeddingRequest):
    return {
        "score": service.compute_similarity(request.sentences)
    }


uvicorn.run(app, host="0.0.0.0", port=8000)
