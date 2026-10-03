# Hugging Face Spaces Deployment Guide

## Setup

1. Create a new Space at https://huggingface.co/spaces
2. Choose **Streamlit** as the SDK
3. Upload your project files or connect via Git

## Important Resource Considerations

- HF Spaces free tier: **2 CPU cores, 16 GB RAM**
- Avoid loading both DistilBERT and BERT simultaneously
- Use `@st.cache_resource` for model loading
- Consider quantized models (e.g., 8-bit) for memory efficiency

## app.py adjustments for HF Spaces

```python
# Load only the smallest/best model at startup
# Use CPU inference (no GPU on free tier)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

## Recommended Model Storage

Store large model files in the HF Model Hub, not in the Space repo:

```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="your-username/airline-models",
                               filename="severity_model.pkl")
```

## requirements.txt for HF Spaces

Same as project `requirements.txt`, minus `boto3`, `psycopg2-binary`.
