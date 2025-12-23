# Indent Discovery V2 (Conversation-Driven Intent/FAQ/Blueprint Extraction)

Streamlit app to upload conversation transcripts and run a staged pipeline:
Normalize → Atoms → FAQs → Intents/Flows → Blueprint.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
```

## Run
```bash
streamlit run streamlit_app.py
```

## Notes
- Artifacts are stored under `workspace/<project_id>/`.
- LLM calls are cached under `workspace/<project_id>/cache/`.
- Embeddings fall back to deterministic hashing if no API key is provided.
