"""
api.py - API FastAPI du pipeline OCR DPCR
GET  /           Interface web custom
GET  /health     Vérification serveur
POST /extraire   Détection auto
POST /extraire/permis
POST /extraire/carte-grise
"""

import os, uuid, tempfile, logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse

from config          import GROQ_API_KEY
from pipeline        import extraire_document, extraire_permis, extraire_carte_grise
from supabase_client import sauvegarder_document

logger = logging.getLogger(__name__)
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] Chargement des modèles OCR fallback...")
    try:
        from ocr_fallback import charger_modeles
        charger_modeles()
        print("[api] Modèles OCR prêts")
    except Exception as e:
        print(f"[api] Modèles non chargés : {e}")
    yield

app = FastAPI(title="API OCR DPCR",
              description="Pipeline OCR pour permis de conduire et certificats d'immatriculation djiboutiens.",
              version="2.0", lifespan=lifespan)

def _save(f: UploadFile) -> str:
    ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
    p = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{ext}")
    with open(p, "wb") as out: out.write(f.file.read())
    return p

def _clean(p):
    try:
        if os.path.exists(p): os.unlink(p)
    except: pass

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def interface():
    if TEMPLATE_PATH.exists():
        return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Interface introuvable</h1>")

@app.get("/health", summary="Vérification serveur")
def health():
    return {"status":"ok","version":"2.0","framework":"fastapi",
            "groq":"configuré" if GROQ_API_KEY else "manquant"}

@app.post("/extraire", summary="Extraction automatique (détection du type)")
def extraire_auto(image: UploadFile = File(...)):
    p = _save(image)
    try:
        r = extraire_document(p)
        if r.get("succes"): sauvegarder_document(r)
        return JSONResponse(r)
    except Exception as e:
        return JSONResponse({"succes":False,"erreur":str(e)},status_code=500)
    finally: _clean(p)

@app.post("/extraire/permis", summary="Force le pipeline permis de conduire")
def extraire_permis_ep(image: UploadFile = File(...)):
    p = _save(image)
    try:
        r = extraire_permis(p)
        if r.get("succes"): sauvegarder_document(r)
        return JSONResponse(r)
    except Exception as e:
        return JSONResponse({"succes":False,"erreur":str(e)},status_code=500)
    finally: _clean(p)

@app.post("/extraire/carte-grise", summary="Force le pipeline certificat d'immatriculation")
def extraire_cg_ep(image: UploadFile = File(...)):
    p = _save(image)
    try:
        r = extraire_carte_grise(p)
        if r.get("succes"): sauvegarder_document(r)
        return JSONResponse(r)
    except Exception as e:
        return JSONResponse({"succes":False,"erreur":str(e)},status_code=500)
    finally: _clean(p)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[api] Interface  : http://localhost:{port}")
    print(f"[api] Swagger    : http://localhost:{port}/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=port,
                reload=os.environ.get("DEBUG","false").lower()=="true")
