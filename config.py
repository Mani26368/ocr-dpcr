"""
config.py
---------
Chargement des clés API et configuration globale du pipeline.
Les clés ne sont JAMAIS codées en dur ici — elles viennent
des variables d'environnement ou du fichier .env.

Usage :
    from config import GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY, TESSERACT_LANG
"""

import os

# ── Chargement .env si python-dotenv est installé ────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optionnel — les variables peuvent venir directement de l'env


# ── Clés API ─────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY  = os.environ.get("SUPABASE_KEY", "")

if not GROQ_API_KEY:
    print("[config] GROQ_API_KEY manquante — Groq Vision désactivé")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("[config] Supabase non configuré — persistance désactivée")


# ── Tesseract ────────────────────────────────────────────────────────────────
try:
    import pytesseract
    _langs = pytesseract.get_languages(config="")
    TESSERACT_LANG = "fra" if "fra" in _langs else "eng"
except Exception:
    TESSERACT_LANG = "eng"

# ── Dossier temporaire pour les images intermédiaires ────────────────────────
import tempfile
TMP_DIR = tempfile.gettempdir()
