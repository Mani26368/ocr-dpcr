# API OCR DPCR — FastAPI

## Lancement (Windows)

```cmd
cd C:\Users\makom\Downloads\dpcr_fastapi

pip install fastapi uvicorn[standard] python-multipart python-dotenv requests

copy .env.example .env
notepad .env

python api.py
```

Ouvrir dans le navigateur :
- http://localhost:8000/health
- http://localhost:8000/docs   ← Swagger UI pour tester tous les endpoints

## Endpoints

| Méthode | URL                   | Description                   |
|---------|-----------------------|-------------------------------|
| GET     | /health               | Vérification serveur          |
| POST    | /extraire             | Détection auto + extraction   |
| POST    | /extraire/permis      | Pipeline permis forcé         |
| POST    | /extraire/carte-grise | Pipeline carte grise forcé    |
