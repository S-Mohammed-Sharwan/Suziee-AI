# Suziee AI — local-first desktop assistant

A runnable starter project for a customizable AI assistant. It includes a React UI, FastAPI backend, Ollama model runtime, persistent chat history, image understanding, document upload/Q&A, optional web search, browser voice input/output, wake-word listening for “Suziee”, allowlisted desktop app/site opening, text-only task runs, and a LoRA fine-tuning script.

> This is a starter codebase, not a frontier model or a production-secure multi-user service. Desktop control is deliberately limited to opening WhatsApp Web, Canva, Chrome, or Google; arbitrary shell commands, account access, sending messages, and destructive actions are not enabled. The web search integration is best-effort. PDF Q&A currently uses a bounded document context, not a full vector database/RAG pipeline.

## Architecture

- `frontend/`: React + Vite UI
- `backend/app/`: Firebase-token-protected FastAPI model routes
- Firebase Authentication and Firestore store account-scoped conversations, document text, tasks, and preferences; Firebase Storage stores uploaded images
- `backend/data/`: legacy local JSON files; current authenticated app flows do not use them for user data
- `backend/training/`: example JSONL and LoRA fine-tuning script

## Requirements

- Python 3.10 or 3.11 recommended
- Node.js 20+ and npm
- Ollama installed: https://ollama.com/download
- For local inference, memory depends on model size. Start with `qwen2.5:3b`; vision model `llava:7b` may require significant RAM/VRAM. If your GPU has 4 GB VRAM, use a smaller quantized model and expect CPU/RAM offloading or slower performance.

## Run locally (Windows PowerShell)

### 1. Install and start Ollama

Install Ollama, then in PowerShell:

```powershell
ollama pull qwen2.5:3b
ollama pull llava:7b
ollama list
```

Ollama normally runs as a background service. If needed, start it with `ollama serve` in a separate terminal.

### 2. Start the backend

```powershell
cd backend
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend docs: http://127.0.0.1:8000/docs
Health check: http://127.0.0.1:8000/api/health

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate again.

### 3. Start the frontend (new terminal)

```powershell
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite, normally http://localhost:5173.

## Run locally (macOS/Linux)

Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend, in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

## Features

- **Chat:** `/api/chat`, with account-scoped conversation history in Firestore.
- **Vision:** `/api/vision`, calls the configured Ollama vision model.
- **Documents:** upload PDF/TXT/MD/CSV/JSON, store text chunks in Firestore, and ask questions about them.
- **Web search:** opt in using the Web search toggle; DuckDuckGo search results are supplied as untrusted context.
- **Voice:** browser Web Speech API for speech input and speech synthesis for read-aloud. Browser/OS support varies.
- **Wake word:** click **Enable Suziee**, grant microphone permission, keep the Suziee page open, then say “Suziee” followed by a command or question. Example: “Suziee, open Canva” or “Suziee, what is a transformer model?”
- **Desktop actions:** the local backend can open WhatsApp Web, Canva, Chrome, or Google from explicit allowlisted “open …” commands. The backend rejects non-local requests.
- **Tasks:** store tasks and results in Firestore; model runs are text-only and do not execute code or external account actions.
- **Model customization:** LoRA fine-tuning example in `backend/training/train_lora.py`.

## Suziee wake word and desktop controls

- Wake-word listening works while the web app is open in a supported browser (Chrome is recommended), the microphone permission is granted, and the computer is awake. It is not an OS-level always-on service and may stop when the tab/browser closes or the computer sleeps.
- Browser speech recognition may use browser/OS/cloud speech services; wake-word recognition is not guaranteed to be fully offline.
- Run the backend on the same computer whose apps you want to open. Keep it bound to `127.0.0.1` as shown in the local setup. If you deploy the backend to a cloud server, desktop actions will target the cloud server, not your laptop.
- The desktop endpoint has a strict allowlist. Do not add a general shell-execution endpoint or execute raw LLM output as a command.

## Change models

Set environment variables before starting the backend, or edit `backend/.env.example` and export the variables in your shell. For example:

```powershell
$env:CHAT_MODEL="qwen2.5:3b"
$env:VISION_MODEL="llava:7b"
uvicorn app.main:app --reload --port 8000
```

Only use model weights whose license is suitable for your intended personal or commercial use. Model names in Ollama must already be pulled.

## Fine-tune a model (optional)

Fine-tuning is separate from serving the Ollama model. Use a compatible GPU and start with a small dataset. Review the base model license and never put secrets or private user data into training data.

```powershell
cd backend
py -3.10 -m venv .trainenv
.\.trainenv\Scripts\Activate.ps1
pip install -r training-requirements.txt
python training\train_lora.py --data training\example.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --epochs 1 --out training\output
```

The script saves a **LoRA adapter**; it does not automatically convert the result to an Ollama model. To serve the tuned model, you need a compatible merge/export/quantization workflow and a corresponding Ollama Modelfile. Training may fail or run slowly on low-memory hardware. The 0.5B example is deliberately small; the provided examples are demonstrations, not enough to meaningfully improve a model.

Expected JSONL format (one object per line):

```json
{"text":"### User: Hi\n### Assistant: Hello!"}
```

## Deploy the frontend to Vercel

The Vercel project configuration is in `frontend/vercel.json`. Import the repository in Vercel and set the project **Root Directory** to `frontend`; Vercel will use `npm ci`, `npm run build`, and `dist` automatically. Add `VITE_API_URL` as an environment variable in Vercel, pointing to the HTTPS origin of your deployed FastAPI backend, then redeploy. Do not include a trailing slash. The frontend uses `localhost:8000` only during local development; a deployed browser will never try to call the visitor's localhost.

Vercel hosts the React interface, not the Python/Ollama model API. Deploy `backend/` to a secured VM/container that can reach Ollama, run it behind HTTPS, and set `OLLAMA_HOST`, `CHAT_MODEL`, `VISION_MODEL`, `CORS_ORIGINS`, and `FIREBASE_PROJECT_ID` there. Set `FIREBASE_SERVICE_ACCOUNT_JSON` only on the backend using a private Firebase service account. Never put service-account credentials in Vercel `VITE_*` variables or the browser bundle. Desktop actions run on the backend host, not your laptop, when the backend is remote.

## Firebase setup

1. Create a Firebase project and register a Web App. Enable **Authentication → Email/Password**, create a **Cloud Firestore** database, and enable **Cloud Storage**.
2. Copy `frontend/.env.example` to `frontend/.env`, then fill in the Firebase Web App config. These web config values are public identifiers; Firestore and Storage rules enforce user access.
3. Deploy `frontend/firestore.rules` and `frontend/storage.rules` from the `frontend/` directory with the Firebase CLI (`firebase deploy --only firestore:rules,storage`). The rules restrict each account to its own documents and image paths.
4. Create a Firebase Admin service account. Put its JSON in `FIREBASE_SERVICE_ACCOUNT_JSON` in `backend/.env` for local development, or in your backend host's secret manager for deployment. Set `FIREBASE_PROJECT_ID` to the same project ID.
5. Set `VITE_API_URL` in `frontend/.env` to the backend origin for local use and in Vercel to the deployed HTTPS backend origin. Set backend `CORS_ORIGINS` to the exact frontend origin.
6. Install backend dependencies again after updating requirements, then restart Vite and FastAPI. Create an account on Suziee’s sign-in page.

Conversations and messages, document text chunks, tasks and results, account theme preferences, and image metadata are stored under `users/{uid}` in Firestore. Uploaded images are stored under the matching user path in Firebase Storage. The model backend verifies Firebase ID tokens and processes requests without saving user chat/document/task data to its local JSON files. Existing local JSON history is not automatically migrated.

Keep the backend private behind HTTPS, verify Firebase ID tokens on every user-specific API, and keep the service-account key secret. Vercel's serverless filesystem is not persistent and should not be used for user data.

## Troubleshooting

- **Backend says Ollama unavailable:** make sure Ollama is running, `ollama list` works, and the model name is pulled.
- **Vision fails:** pull `llava:7b`; on low-memory hardware, use a smaller supported vision model and update `VISION_MODEL`.
- **CORS error:** set `CORS_ORIGINS` to the exact frontend origin, restart backend.
- **Web search fails:** network/DDG availability can vary; turn web search off and use local chat.
- **Training OOM:** lower `--max-length`, use a smaller base model, close other GPU tasks, or use a cloud GPU.
- **Voice not available:** try a supported browser such as Chrome; browser speech services may use OS/cloud services and are not guaranteed fully offline.

## API summary

- `GET /api/health`
- `GET /api/models`
- `GET /api/conversations` (legacy; account data is loaded from Firestore)
- `GET /api/conversations/{conversation_id}` (legacy)
- `POST /api/chat` (Firebase ID token required; stateless model response)
- `POST /api/vision` (multipart file + prompt)
- `POST /api/documents` (multipart file; extracts text for frontend Firestore storage)
- `GET /api/documents`
- `POST /api/documents/ask`
- `GET /api/tasks` (legacy; tasks are loaded from Firestore)
- `POST /api/tasks/run` (Firebase ID token required; stateless task result)
- `GET /api/desktop/status`
- `POST /api/desktop/command` (local-only, allowlisted open actions)

## Security notes

The UI requires Firebase Authentication before showing personal data. Apply the included Firestore and Storage rules before using real accounts. The FastAPI model endpoints require Firebase ID tokens; only health status is public. Keep service-account JSON on the backend. Never let an LLM-generated string be executed directly as a shell command. Desktop control remains allowlisted and local-only.
