"""
groq_extraction.py
------------------
Extraction des champs via Groq Vision (llama-4-scout).
Moteur PRINCIPAL pour les deux types de documents.

Fonctions exposées :
  - extraire_permis_groq(img_path, api_key)      -> dict | None
  - extraire_carte_grise_groq(img_path, api_key) -> dict | None
  - detecter_type_document(img_path, api_key)    -> 'permis' | 'carte_grise' | None
"""

import base64
import json
import re
import cv2
import os

from groq import Groq
from pretraitement import normaliser_pour_groq


MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _encode_image(img_path: str) -> tuple[str, str]:
    """Encode une image en base64. Retourne (b64_string, media_type)."""
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    media_type = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
    return b64, media_type


def _preparer_image(img_path: str, tmp_dir: str = ".") -> str:
    """
    Normalise l'image (luminosité, débruitage) avant envoi à Groq.
    Retourne le chemin de l'image normalisée.
    """
    img = cv2.imread(img_path)
    if img is None:
        return img_path
    img = normaliser_pour_groq(img)
    base, ext = os.path.splitext(img_path)
    norm_path = base + "_norm" + ext
    cv2.imwrite(norm_path, img)
    return norm_path


def _nettoyer_json(raw: str) -> str:
    """
    Supprime les backticks et tout texte précédant le JSON.
    Gère les variantes : ``` , ```json , ```python ...
    """
    raw = re.sub(r"^```[\w]*\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw).strip()
    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        raw = match.group(0)
    return raw


# ── Détection du type de document ────────────────────────────────────────────

def detecter_type_document(img_path: str, api_key: str) -> str | None:
    """
    Détecte si l'image est un permis de conduire ou un certificat d'immatriculation.
    Retourne 'permis', 'carte_grise', ou None si la détection échoue.
    """
    if not api_key:
        return None

    b64, media_type = _encode_image(img_path)
    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": (
                    "Regarde cette image de document djiboutien.\n"
                    "Est-ce un PERMIS DE CONDUIRE ou un CERTIFICAT D'IMMATRICULATION (carte grise) ?\n"
                    "Réponds UNIQUEMENT avec : permis  OU  carte_grise\n"
                    "Aucun autre mot."
                )}
            ]}],
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip().lower()
        if "carte" in raw or "immatriculation" in raw or "grise" in raw:
            return "carte_grise"
        return "permis"
    except Exception as e:
        print(f"  [groq] Détection type échouée : {e}")
        return None


# ── Extraction permis ─────────────────────────────────────────────────────────

PROMPT_PERMIS = """Tu es un expert en lecture de permis de conduire djiboutiens.

STRUCTURE DU PERMIS DJIBOUTIEN :
  Partie GAUCHE :
    - Ligne 1 'Nom' : nom complet du titulaire (souvent sur 2 lignes)
    - Ligne 2 'Date et lieu de naissance' : date JJ/MM/AAAA + ville
    - Ligne 3 'Domicile' : adresse (ex: CITE BARWAKO, BALBALA, DJIBOUTI)
    - En bas : date de délivrance + numéro de permis format NNNN/AA

  Partie DROITE (tableau catégories A1, A, B, C, D, E, F) :
    - Catégorie ACTIVE = ligne avec une date + un numéro + tampon DMSR
    - Les labels A, B, C à gauche du tableau NE SONT PAS des catégories actives

RÈGLES STRICTES :
  - nom : UNIQUEMENT le nom de la personne, jamais NOM/PERMIS/SCEAU/TEMPORAIRE
  - date_naissance : format JJ/MM/AAAA
  - domicile : adresse après 'Domicile' — si DJIBOUTI, mettre DJIBOUTI
  - numero_permis : format NNNN/AA (ex: 4890/20, 5274/22)
  - categorie : UNIQUEMENT la lettre de la ligne active (ex: B)

EXEMPLES RÉELS :
  {"nom":"RAMADAN ISSA ABDILLAHI","date_naissance":"30/11/2000","lieu_naissance":"DJIBOUTI",
   "domicile":"CITE BARWAKO","numero_permis":"4890/20","categorie":"B"}
  {"nom":"SAAD IBRAHIM FARAH","date_naissance":"04/10/2002","lieu_naissance":"DJIBOUTI",
   "domicile":"DJIBOUTI","numero_permis":"5274/22","categorie":"B"}

RÉPONDS UNIQUEMENT avec le JSON, sans backticks, sans texte autour :
{"nom":"...","date_naissance":"...","lieu_naissance":"...","domicile":"...","numero_permis":"...","categorie":"..."}"""


def extraire_permis_groq(img_path: str, api_key: str) -> dict | None:
    """
    Extrait les 6 champs du permis djiboutien via Groq Vision.
    Retourne un dict ou None si l'extraction échoue.
    """
    if not api_key:
        return None

    img_path = _preparer_image(img_path)
    b64, media_type = _encode_image(img_path)
    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": PROMPT_PERMIS},
            ]}],
            max_tokens=300,
        )

        raw  = response.choices[0].message.content.strip()
        raw  = _nettoyer_json(raw)
        data = json.loads(raw)

        # Post-traitement
        if data.get("nom"):
            data["nom"] = str(data["nom"]).upper().strip()

        if data.get("date_naissance"):
            m = re.search(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", str(data["date_naissance"]))
            if m:
                data["date_naissance"] = f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/{m.group(3)}"

        if data.get("numero_permis"):
            m = re.search(r"(\d{3,5})[/\-](\d{2})", str(data["numero_permis"]))
            if m:
                data["numero_permis"] = f"{m.group(1)}/{m.group(2)}"

        if data.get("categorie"):
            cats_valides = {"A","A1","A2","B","B1","BE","C","C1","CE","C1E","D","D1","DE","D1E","F","G"}
            tokens = re.findall(r"\b[A-Z][A-Z0-9]{0,2}\b", str(data["categorie"]).upper())
            cats = [t for t in tokens if t in cats_valides]
            data["categorie"] = ", ".join(cats) if cats else None

        for field in ("lieu_naissance", "domicile"):
            if data.get(field):
                data[field] = str(data[field]).upper().strip()

        # Nettoyage valeurs null chaîne
        for k in data:
            if data[k] in ("null", "None", ""):
                data[k] = None

        return data

    except json.JSONDecodeError as e:
        print(f"  [groq] JSON invalide permis : {e} | brut : {raw[:150]}")
        return None
    except Exception as e:
        print(f"  [groq] Erreur permis : {e}")
        return None


# ── Extraction carte grise ────────────────────────────────────────────────────

PROMPT_CARTE_GRISE = """Tu es un expert en lecture de certificats d'immatriculation djiboutiens.

Ce document peut être orienté à 90° ou 180° — lis-le dans tous les sens si nécessaire.

CHAMPS À EXTRAIRE :
(A) NUMERO D'IMMATRICULATION     -> "immatriculation"   ex: 419D93
(B) DATE DE 1ère MISE EN CIRC.   -> "date_mise_en_circ" ex: 11/05/2022
(C) Nom et prénom du propriétaire -> "nom"              ex: ABDIRAHMAN AHMED ABDILLAHI
(E) Domicile                     -> "domicile"          ex: CITE-GACHAMALEH
(F) Marque du véhicule           -> "marque"            ex: MITSUBISHI
    Modèle / Type                -> "modele"            ex: L200
    Energie                      -> "energie"           ex: DIESEL
(G) N° immat. précédent ou série -> "numero_serie"      ex: MMBNGV548NH007212
    Profession du propriétaire   -> "profession"        ex: INGENIEUR (optionnel)

RÈGLES :
- immatriculation (A) est COURT : ex: 419D93
- numero_serie (G) est LONG : 17+ caractères
- Si un champ est absent, mets null

RÉPONDS UNIQUEMENT avec le JSON valide, sans backticks :
{"immatriculation":"...","date_mise_en_circ":"...","nom":"...","domicile":"...",
 "marque":"...","modele":"...","energie":"...","numero_serie":"...","profession":"..."}"""


def extraire_carte_grise_groq(img_path: str, api_key: str) -> dict | None:
    """
    Extrait les 9 champs du certificat d'immatriculation via Groq Vision.
    Retourne un dict ou None si l'extraction échoue.
    """
    if not api_key:
        return None

    img_path = _preparer_image(img_path)
    b64, media_type = _encode_image(img_path)
    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": PROMPT_CARTE_GRISE},
            ]}],
            max_tokens=400,
        )

        raw  = response.choices[0].message.content.strip()
        raw  = _nettoyer_json(raw)
        data = json.loads(raw)

        # Post-traitement
        if data.get("immatriculation"):
            data["immatriculation"] = str(data["immatriculation"]).upper().replace(" ", "")

        if data.get("date_mise_en_circ"):
            m = re.search(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", str(data["date_mise_en_circ"]))
            if m:
                data["date_mise_en_circ"] = f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/{m.group(3)}"

        if data.get("nom"):
            data["nom"] = str(data["nom"]).upper().strip()

        for field in ("marque", "modele", "energie"):
            if data.get(field):
                data[field] = str(data[field]).upper().strip()

        # Nettoyage valeurs null chaîne
        for k in data:
            if data[k] in ("null", "None", ""):
                data[k] = None

        return data

    except json.JSONDecodeError as e:
        print(f"  [groq] JSON invalide carte grise : {e} | brut : {raw[:150]}")
        return None
    except Exception as e:
        print(f"  [groq] Erreur carte grise : {e}")
        return None
