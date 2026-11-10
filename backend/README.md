# Suziee AI API (Gemini on Vercel)

This folder is the FastAPI backend deployment root. It can run as a separate Vercel project from the existing Vite frontend.

## Vercel project settings

- Connect the `Suziee-AI` GitHub repository.
- Set the project Root Directory to `backend`.
- Keep the framework preset as FastAPI (or Other if Vercel does not detect it automatically).
- Add these server-side environment variables for Production and Preview:
  - `AI_PROVIDER=gemini`
  - `GEMINI_API_KEY` (create a key in Google AI Studio; add it directly in Vercel, never in the frontend)
  - `GEMINI_MODEL=gemini-3.8-flash`
  - `GEMINI_VISION_MODEL=gemini-3.8-flash`
  - `FIREBASE_PROJECT_ID=suziee-ai`
  - `FIREBASE_SERVICE_ACCOUNT_JSON` (Firebase Admin service account JSON; keep secret)
  - `CORS_ORIGINS=https://suziee-ai.vercel.app`
- Deploy. The API health endpoint is `/api/health`.
- Set the frontend project's `VITE_API_URL` to the new API project's HTTPS domain, then redeploy the frontend.

The backend verifies Firebase ID tokens before private endpoints and only sends prompts to Gemini from the server. The Gemini key must never use a `VITE_` prefix or be placed in frontend variables.

Google AI Studio's free tier has model-specific quotas and may change. The free tier's terms allow submitted content to be used to improve Google's products, so do not send sensitive or confidential data. Vercel Functions have request-size and duration limits; keep image/document requests small and check current Vercel limits for the plan.

Local development remains supported with `AI_PROVIDER=ollama`, `OLLAMA_HOST`, `CHAT_MODEL`, and `VISION_MODEL` in `backend/.env`.
