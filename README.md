# DocChat on Render

Files needed: `app.py` + `requirements.txt` (+ `render.yaml` for one-click setup).

1. Push this folder to a GitHub repo (`.env` is ignored by `.gitignore`).
2. Render dashboard → New → Blueprint → pick the repo → Apply.
3. When asked, paste `OPENAI_API_KEY`.
4. Open the `.onrender.com` URL when the deploy is Live.

Without the Blueprint: New → Web Service →
Build: `pip install -r requirements.txt`
Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
Env vars: `OPENAI_API_KEY`, `PYTHON_VERSION=3.12.8`.

Free-plan limits: sleeps after ~15 min idle (about a minute to wake), 512 MB RAM,
no persistent disk — the document index lives in memory and is lost on restart.
