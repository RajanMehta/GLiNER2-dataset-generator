# GLiNER2 Dataset Generator

A Streamlit app for generating JSONL training datasets for finetuning [GLiNER2](https://github.com/urchade/GLiNER) models. Define an annotation schema, describe your domain, and export ready-to-use training data. Powered by OpenAI or Ollama.

## Features

- **Three task types:** Named Entity Recognition (NER), Classification (CLF), JSON Extraction — combinable in a single dataset
- **Provider support:** OpenAI API or hosted/local Ollama
- **Validation:** hard excludes for structural failures, soft warnings for value mismatches
- **Batching:** automatically splits large requests into batches of 25


## Developer Setup

**Requirements:** Python 3.9+

```bash
make env          # creates .venv and installs requirements.txt
make up           # starts streamlit in background
make down         # stops streamlit
```

The app runs at `http://localhost:8501` by default.

To install additional packages:
```bash
.venv/bin/pip install <package>
```

## LLM Provider Configuration

| Provider | Base URL | API Key |
|---|---|---|
| OpenAI | _(managed by SDK)_ | OpenAI API key |
| Ollama (hosted) | `https://ollama.com` | Bearer token from ollama.com |
| Ollama (local) | `http://localhost:11434` | _(not required)_ |

## Output Format

Each line in the exported `.jsonl` file follows the GLiNER2 training format:

```json
{"input": "What's my savings account balance?", "output": {
  "entities": {"account_type": ["savings account"]},
  "classifications": [{"task": "intent", "labels": ["check_balance", "make_transfer"], "true_label": ["check_balance"]}]
}}
```

- See [train_data.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/8-train_data.md) for the full schema reference.
- See [training.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md) to learn more about fine-tuning GLiNER2.
