import os
from sentence_transformers import SentenceTransformer, util

model: SentenceTransformer = SentenceTransformer(os.environ.get('MODEL_PATH'))


def encode(sentences: str | list[str]):
    if isinstance(sentences, str):
        sentences = [sentences]
    data = model.encode(sentences).tolist()
    return [{"object": "embedding", "embedding": e, "index": i} for i, e in enumerate(data)]


def compute_similarity(sentences: list[str]):
    embeddings = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(embeddings, embeddings)
    resp = []
    for i in range(len(sentences)-1):
        for j in range(i+1, len(sentences)):
            resp.append(
                {"sentence1": sentences[i],
                 "sentence2": sentences[j],
                 "score": scores[i][j].item()})
    return resp
