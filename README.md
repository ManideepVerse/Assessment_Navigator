# SHL Assessment Recommender

A conversational AI agent built with **FastAPI**, **FAISS**, and **Google Gemini** that helps recruiters and hiring managers find the perfect SHL individual test solutions. 

This project was built to solve the ambiguity in the hiring process. Rather than forcing users to guess the right vocabulary or navigate complex keyword filters, this stateless agent accepts vague intents (e.g., *"I need to hire a Java developer"*), asks targeted clarifying questions, and recommends highly-relevant, data-grounded assessments directly from the SHL catalog.

## Architecture & Tech Stack

- **Backend Framework:** FastAPI (Python)
- **Large Language Model:** Google Gemini (`gemini-flash-latest`) via `google-generativeai`
- **Vector Database:** FAISS (in-memory) for ultra-fast, stateless semantic retrieval.
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Deployment:** Render (`render.yaml` provided)

## Core Features

- **Stateless API (`/chat`)**: The backend stores no session data. The entire conversational context is passed in the request payload, making the API infinitely horizontally scalable.
- **Grounded Recommendations**: The agent strictly uses RAG tool-calling to fetch live assessment data. It mathematically cannot hallucinate URLs or invent fake assessments.
- **Strict Schema Compliance**: Responses strictly adhere to a deterministic JSON schema (with a `reply`, `recommendations` array, and `end_of_conversation` flag).
- **Conversational Refinement**: The agent gracefully handles constraints being changed mid-conversation (e.g., *"Actually, add a personality test too"*) without restarting the flow.

## Getting Started Locally

### Prerequisites
- Python 3.10+
- A Google Gemini API Key

### Installation

1. Clone the repository and navigate into it:
   ```bash
   git clone <your-repo-url>
   cd SHL_Task
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup your environment variables by creating a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   > **⚠️ IMPORTANT:** You MUST replace `your_gemini_api_key_here` with your actual Google Gemini API Key before running the server! Do not share this key publicly.

### Rebuilding the Index (Optional)
The project comes pre-packaged with the `catalog.faiss` and `catalog_metadata.pkl` files. If the SHL catalog updates in the future, you can rebuild the vector database by running:
```bash
python ingest_data.py
python build_index.py
```

### Running the Server
Start the FastAPI server via Uvicorn:
```bash
python main.py
```
The API will be available at `http://localhost:8000`.

## API Endpoints

### `GET /health`
Returns the readiness state of the API.
**Response:** `{"status": "ok"}`

### `POST /chat`
Accepts a stateless conversation history and returns the agent's next action.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a senior backend developer"}
  ]
}
```

**Response:**
```json
{
  "reply": "Could you clarify what specific programming languages or frameworks they will be using?",
  "recommendations": [],
  "end_of_conversation": false
}
```

## Deployment to Render

This project includes a `render.yaml` file for instant 1-click deployment.

1. Push this repository to GitHub.
2. Log into [Render](https://render.com) and create a **New Blueprint Instance** (or Web Service).
3. Connect your GitHub repository.
4. Render will automatically detect the Python environment and the start command (`uvicorn main:app --host 0.0.0.0 --port $PORT`).
5. **Important:** Navigate to your service settings in Render and add `GEMINI_API_KEY` to the Environment Variables.

