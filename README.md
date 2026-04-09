# AI Assistant

A local AI assistant inspired by Iron Man's Jarvis — powered by Ollama, LangChain, and local LLMs. It runs entirely on your machine with tool-augmented reasoning, memory, and multi-agent capabilities.

---

## Features

- Conversational AI with persistent memory
- Web search via Google Custom Search API
- Weather & air quality (OpenWeatherMap + AccuWeather)
- PDF Q&A using RAG (FAISS + HuggingFace embeddings)
- Deep web research with multi-step reasoning
- Terminal command execution (local + SSH)
- Code generation via StarCoder2
- Image analysis via LLaVA
- Alarm scheduling and Google Calendar integration
- System status monitoring (CPU, RAM, disk)
- Text-to-speech via ElevenLabs / gTTS

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the following models:
  - `gemma2:2b` — main agent LLM
  - `jarvis:latest` — custom fine-tuned model
  - `deepseek-r1:7b` — deep reasoning
  - `starcoder2:7b` — code generation
  - `llava:7b` — image analysis
  - `wizardcoder` — terminal command generation
- `sshpass` (for SSH tool)
- `tesseract-ocr` (for OCR on scanned PDFs)
- A local LLaMA `.gguf` model file (for `LlamaCpp`)

---

## Setup

```bash
pip install -r requirements.txt
```

Update the following in `main.py` and `Tools.py`:
- `CONFIG["API_KEY"]` — Google Custom Search API key
- `CONFIG["CSE_ID"]` — Google Custom Search Engine ID
- `CONFIG["WEATHER_API_KEY"]` — OpenWeatherMap API key
- `ACCUWEATHER_API_KEY` — AccuWeather API key
- `model_path` in `initialize_agent_system()` — path to your `.gguf` model

For Google Calendar, place your OAuth client secret JSON in the project root and update `CLIENT_SECRET_FILE` in `calendar_agent.py`.

---

## Usage

```bash
python main.py
```

**Commands inside the assistant:**

| Command  | Description                        |
|----------|------------------------------------|
| `/exit`  | Quit the assistant                 |
| `/save`  | Manually save conversation history |
| `/load`  | Reload previous conversation       |
| `/help`  | Show available commands            |

---

## Project Structure

```
├── main.py              # Entry point, agent initialization
├── Tools.py             # All tool implementations
├── RAG.py               # PDF ingestion and Q&A chain
├── DeepSearch.py        # Multi-step deep web research agent
├── alarm_agent.py       # Alarm scheduling
├── calendar_agent.py    # Google Calendar integration
├── AgentEmail.py        # Email agent (Gmail)
├── OpenCV.py            # Computer vision utilities
├── PC_records.py        # PC usage tracking
├── Testors.py           # Testing utilities
├── pdf_vectorstore/     # FAISS vector store for PDFs
└── chat_history.json    # Persisted conversation history
```

---

## Notes

- All LLM inference runs locally via Ollama — no data leaves your machine (except API calls to weather/search services).
- PDF documents are loaded from `/home/jarvis/Documents/` by default. Update `DOCUMENTS_DIR` in `RAG.py` to change this.
- SSH tool connects to `192.168.1.2:8022` by default. Update credentials in `Tools.py`.
