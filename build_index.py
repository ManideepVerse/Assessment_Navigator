import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def build_index():
    print("Loading catalog...")
    with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
        catalog = json.loads(f.read(), strict=False)

    documents = []
    metadata = []

    for item in catalog:
        name = item.get("name", "")
        desc = item.get("description", "") or ""
        keys = ", ".join(item.get("keys", []))
        job_levels = ", ".join(item.get("job_levels", []))
        duration = item.get("duration", "") or ""
        languages = ", ".join(item.get("languages", [])[:5])

        # Rich text for embedding — includes all searchable fields
        doc_text = (
            f"Name: {name}\n"
            f"Categories: {keys}\n"
            f"Job Levels: {job_levels}\n"
            f"Duration: {duration}\n"
            f"Languages: {languages}\n"
            f"Description: {desc}"
        )
        documents.append(doc_text)

        # Store ALL raw data for the agent to use
        metadata.append({
            "name": name,
            "url": item.get("link", ""),
            "description": desc,
            "keys": item.get("keys", []),
            "job_levels": item.get("job_levels", []),
            "duration": duration,
            "languages": item.get("languages", []),
            "remote": item.get("remote", ""),
            "adaptive": item.get("adaptive", ""),
        })

    print(f"Loaded {len(documents)} documents. Initializing SentenceTransformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "catalog.faiss")
    with open("catalog_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Done! Index has {index.ntotal} vectors, dimension={dimension}.")

if __name__ == "__main__":
    build_index()
