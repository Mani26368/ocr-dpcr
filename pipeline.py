"""
pipeline.py
-----------
Fonctions principales du pipeline OCR DPCR.

Point d'entrée unique : extraire_document(image_path)

Fonctions exposées :
  - extraire_document(image_path, type_doc=None) -> dict
  - extraire_permis(image_path)                  -> dict
  - extraire_carte_grise(image_path)             -> dict
"""

import os
import cv2
import tempfile

from config          import GROQ_API_KEY, TMP_DIR
from scanner         import scanner_document
from groq_extraction import (
    detecter_type_document,
    extraire_permis_groq,
    extraire_carte_grise_groq,
)
from ocr_fallback import (
    extraire_texte_tous_moteurs,
    extraire_champs_permis,
    extraire_champs_carte_grise_regex,
)


# ── Pipeline permis ───────────────────────────────────────────────────────────

def extraire_permis(image_path: str) -> dict:
    """
    Pipeline complet pour un permis de conduire djiboutien.

    1. Scanner virtuel (redressement perspective)
    2. Groq Vision -> tous les champs en 1 appel (moteur principal)
    Fallback : OCR cascade (DocTR -> EasyOCR -> Tesseract + regex)

    Retourne un dict JSON :
      type_document, nom, date_naissance, lieu_naissance,
      domicile, numero_permis, categorie, succes, erreur
    """
    try:
        print(f"\n[pipeline] Traitement permis : {os.path.basename(image_path)}")

        # Étape 1 : Scanner
        warped, warped_thresh, warped_path = scanner_document(image_path, tmp_dir=TMP_DIR)

        # Étape 2 : Groq Vision (moteur principal)
        data = extraire_permis_groq(warped_path, GROQ_API_KEY)

        # Fallback OCR si Groq indisponible
        if data is None:
            print("  [pipeline] Groq indisponible -> fallback OCR")
            resultats = extraire_texte_tous_moteurs(warped, warped_thresh, warped_path)
            categorie = None

            champs_d = extraire_champs_permis(resultats["doctr"]["text"],   categorie)
            champs_e = extraire_champs_permis(resultats["easyocr"]["text"], categorie)
            champs_t = extraire_champs_permis(resultats["tesseract"]["text"], categorie)

            champs = champs_d.copy()
            for champ in champs:
                if champs[champ] is None:
                    champs[champ] = champs_e.get(champ) or champs_t.get(champ)

            data = {
                "nom":            champs.get("Nom"),
                "date_naissance": champs.get("Date naissance"),
                "lieu_naissance": champs.get("Lieu naissance"),
                "domicile":       champs.get("Domicile"),
                "numero_permis":  champs.get("N° permis"),
                "categorie":      champs.get("Catégorie"),
            }

        # Nettoyage
        for k in (data or {}):
            if data[k] in ("null", "None", ""):
                data[k] = None

        nb = sum(1 for v in (data or {}).values() if v is not None)

        return {
            "type_document":  "permis",
            **(data or {}),
            "succes": nb >= 3,
            "erreur": None,
        }

    except Exception as e:
        print(f"  [pipeline] Erreur permis : {e}")
        return {
            "type_document": "permis",
            "nom": None, "date_naissance": None, "lieu_naissance": None,
            "domicile": None, "numero_permis": None, "categorie": None,
            "succes": False, "erreur": str(e),
        }


# ── Pipeline carte grise ──────────────────────────────────────────────────────

def extraire_carte_grise(image_path: str) -> dict:
    """
    Pipeline complet pour un certificat d'immatriculation djiboutien.

    1. Scanner virtuel
    2. Groq Vision -> tous les champs en 1 appel (moteur principal)
    Fallback : regex sur texte DocTR

    Retourne un dict JSON :
      type_document, immatriculation, date_mise_en_circ, nom, domicile,
      marque, modele, energie, numero_serie, profession, succes, erreur
    """
    try:
        print(f"\n[pipeline] Traitement carte grise : {os.path.basename(image_path)}")

        # Étape 1 : Scanner
        warped, warped_thresh, warped_path = scanner_document(image_path, tmp_dir=TMP_DIR)

        # Étape 2 : Groq Vision (moteur principal)
        data = extraire_carte_grise_groq(warped_path, GROQ_API_KEY)

        # Fallback regex si Groq indisponible
        if data is None:
            print("  [pipeline] Groq indisponible -> fallback OCR")
            resultats  = extraire_texte_tous_moteurs(warped, warped_thresh, warped_path)
            data       = extraire_champs_carte_grise_regex(resultats["doctr"]["text"])
            data_easy  = extraire_champs_carte_grise_regex(resultats["easyocr"]["text"])
            for champ in data:
                if data[champ] is None and data_easy.get(champ):
                    data[champ] = data_easy[champ]

        # Nettoyage
        for k in (data or {}):
            if data[k] in ("null", "None", ""):
                data[k] = None

        nb = sum(1 for v in (data or {}).values() if v is not None)

        return {
            "type_document": "carte_grise",
            **(data or {}),
            "succes": nb >= 4,
            "erreur": None,
        }

    except Exception as e:
        print(f"  [pipeline] Erreur carte grise : {e}")
        return {
            "type_document": "carte_grise",
            "immatriculation": None, "date_mise_en_circ": None, "nom": None,
            "domicile": None, "marque": None, "modele": None,
            "energie": None, "numero_serie": None, "profession": None,
            "succes": False, "erreur": str(e),
        }


# ── Point d'entrée unique ─────────────────────────────────────────────────────

def extraire_document(image_path: str, type_doc: str = None) -> dict:
    """
    Point d'entrée UNIQUE du pipeline.
    Détecte automatiquement le type de document et appelle le bon pipeline.

    Paramètres
    ----------
    image_path : chemin vers la photo
    type_doc   : forcer 'permis' ou 'carte_grise' (auto-détecté si None)

    Retourne
    --------
    dict JSON avec type_document + tous les champs extraits
    """
    if type_doc is None:
        type_doc = detecter_type_document(image_path, GROQ_API_KEY)

    if type_doc is None:
        # Groq indisponible et pas de type fourni -> défaut permis
        print("  [pipeline] Type non détecté -> permis par défaut")
        type_doc = "permis"

    if type_doc == "carte_grise":
        return extraire_carte_grise(image_path)
    return extraire_permis(image_path)
