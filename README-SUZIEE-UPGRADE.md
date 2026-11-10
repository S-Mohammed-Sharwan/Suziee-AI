# Suziee AI UI and Firebase setup

The interface now includes a Firebase sign-in/sign-up screen, per-account Firestore storage for conversations, document text, tasks, and theme preferences, and Firebase Storage for uploaded images. Assistant code fences render in their own code panels with a separate **Copy code** control. The navigation is fixed as a rounded desktop sidebar and a floating mobile dock.

## Local setup

1. Enable Email/Password in Firebase Authentication. Create Firestore and Storage in the same project.
2. Copy `frontend/.env.example` to `frontend/.env` and enter the Firebase Web App config values.
3. Deploy the account-scoped rules from `frontend/`:

   ```powershell
   firebase deploy --only firestore:rules,storage
   ```

4. Copy `backend/.env.example` to `backend/.env`, set `FIREBASE_PROJECT_ID`, and add the Firebase Admin service account JSON to `FIREBASE_SERVICE_ACCOUNT_JSON`. Keep this private backend-only secret out of frontend environment variables.
5. Reinstall backend requirements and run FastAPI and Vite using the commands in `README.md`.
6. Open the app and create an account.

For Vercel, set the `frontend` directory as the project root and add the `VITE_FIREBASE_*` web config values plus `VITE_API_URL`. Deploy the Python backend separately, set its Firebase Admin secrets there, and configure `CORS_ORIGINS` to the deployed frontend origin. Existing local JSON chat history is not imported automatically.
