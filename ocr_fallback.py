"""
ocr_fallback.py
---------------
Moteurs OCR classiques utilisés UNIQUEMENT en fallback
si Groq Vision est indisponible.

Fonctions exposées :
  - charger_modeles()
  - extraire_texte_tous_moteurs(warped, warped_thresh) -> dict
  - extraire_champs_permis(text, categorie)            -> dict
  - extraire_champs_carte_grise_regex(text)            -> dict
"""

import re
import time
import cv2
import pytesseract
import easyocr

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from config import TESSERACT_LANG

# ── Chargement des modèles (à appeler une seule fois au démarrage) ────────────

_doctr_model   = None
_easyocr_reader = None


def charger_modeles():
    """
    Charge DocTR et EasyOCR en mémoire.
    A appeler une fois au démarrage du serveur.
    """
    global _doctr_model, _easyocr_reader

    print("[ocr] Chargement DocTR...")
    _doctr_model = ocr_predictor(
        det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
    )

    print("[ocr] Chargement EasyOCR...")
    _easyocr_reader = easyocr.Reader(["fr", "en"], gpu=False)

    print("[ocr] Modèles prêts")


def extraire_texte_tous_moteurs(warped, warped_thresh, warped_path: str = "doc_warped.jpg") -> dict:
    """
    Lance les 3 moteurs OCR sur le document redressé.
    Retourne un dict avec les textes de chaque moteur.
    """
    if _doctr_model is None or _easyocr_reader is None:
        raise RuntimeError("Modèles OCR non chargés — appeler charger_modeles() d'abord")

    resultats = {}

    # Tesseract
    ts = time.time()
    resultats["tesseract"] = {
        "text": pytesseract.image_to_string(warped_thresh, lang=TESSERACT_LANG),
        "time": round(time.time() - ts, 2),
    }

    # EasyOCR
    ts = time.time()
    easy_res = _easyocr_reader.readtext(warped)
    resultats["easyocr"] = {
        "text": "\n".join([text for (_, text, _) in easy_res]),
        "raw":  easy_res,
        "time": round(time.time() - ts, 2),
    }

    # DocTR
    ts = time.time()
    doc = DocumentFile.from_images(warped_path)
    res = _doctr_model(doc)
    txt = ""
    for page in res.pages:
        for block in page.blocks:
            for line in block.lines:
                txt += " ".join([w.value for w in line.words]) + "\n"
    resultats["doctr"] = {
        "text": txt,
        "raw":  res,
        "time": round(time.time() - ts, 2),
    }

    print(
        f"  [ocr] Tesseract: {resultats['tesseract']['time']}s | "
        f"EasyOCR: {resultats['easyocr']['time']}s | "
        f"DocTR: {resultats['doctr']['time']}s"
    )
    return resultats


# ── Parsing regex permis ──────────────────────────────────────────────────────

def extraire_champs_permis(text: str, categorie: str | None) -> dict:
    """
    Parse le texte brut OCR et extrait les champs du permis djiboutien.
    Utilisé uniquement en fallback si Groq Vision est indisponible.
    """
    champs = {
        "Nom": None, "Date naissance": None, "Lieu naissance": None,
        "Domicile": None, "N° permis": None, "Catégorie": categorie,
    }

    BLACKLIST_NOM = {
        "SCEAU","NOM","PERMIS","TITRE","CACHET","TEMPORAIRE","PERMANENT",
        "DJIBOUTI","DHBOUT","DUIBOUTT","DJIBOUTT","DELIVRE","AUTORITE",
        "AUTORITÉ","SIGNATURE","DOMICILE","CATEGORIES","VEHICULES","MANENT",
    }
    BLACKLIST_DOMICILE = {
        "TEMPORAIRE","PERMANENT","MANENT","SCEAU","CACHET",
        "SIGNATURE","AUTORITE","AUTORITÉ","TITRE","PERMIS","OU CACHET",
    }
    djibouti_re = r"d[jhu][ih]?b?[oui]*[ut]+"

    def nom_valide(mot):
        mot = mot.strip()
        return (
            mot.upper() not in BLACKLIST_NOM
            and re.match(r"^[A-ZÉÈÀ][A-ZÉÈÀ\s\-]+$", mot)
            and len(mot.split()) <= 4
            and len(mot) >= 3
        )

    lignes = [l.strip() for l in text.split("\n") if l.strip()]

    for i, ligne in enumerate(lignes):
        ll = ligne.lower()

        # Nom
        if re.search(r"\bnom\b", ll) and "domicile" not in ll:
            parties = []
            m = re.search(r"nom\s*[:\-]?\s*i?\s*([A-ZÉÈÀ][A-ZÉÈÀ\s\-]{2,})", ligne, re.IGNORECASE)
            if m and nom_valide(m.group(1)):
                parties.append(m.group(1).strip())
            if not parties:
                for j in range(i - 1, max(i - 4, -1), -1):
                    if nom_valide(lignes[j]):
                        parties.insert(0, lignes[j])
                        break
            for j in range(i + 1, min(i + 6, len(lignes))):
                nxt = lignes[j].strip()
                if nom_valide(nxt) and len(nxt.split()) <= 3 and nxt not in parties:
                    parties.append(nxt)
                    break
            if parties:
                champs["Nom"] = " ".join(parties)

        # Date de naissance
        if re.search(r"naiss|nalss", ll):
            m = re.search(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})", ligne)
            if m:
                champs["Date naissance"] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
            if not champs["Date naissance"]:
                for j in range(i - 1, max(i - 5, -1), -1):
                    m = re.search(r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b", lignes[j])
                    if m:
                        champs["Date naissance"] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
                        break

        # Lieu de naissance
        if re.search(r"naiss|nalss", ll) and not champs["Lieu naissance"]:
            for j in list(range(i - 1, max(i - 5, -1), -1)) + list(range(i + 1, min(i + 6, len(lignes)))):
                if re.search(djibouti_re, lignes[j], re.IGNORECASE):
                    champs["Lieu naissance"] = "DJIBOUTI"
                    break

        # Domicile
        if re.search(r"\bdomicile\b", ll):
            m = re.search(r"domicile\s*[:\-]?\s*([A-ZÉÈÀ][A-Za-zÉÈÀ\.\s]+)", ligne, re.IGNORECASE)
            if m and len(m.group(1).strip()) > 2:
                val = m.group(1).strip()
                champs["Domicile"] = "DJIBOUTI" if re.search(djibouti_re, val, re.IGNORECASE) else val
            else:
                for j in range(i + 1, min(i + 6, len(lignes))):
                    nxt = lignes[j].strip()
                    if nxt.upper() in BLACKLIST_DOMICILE:
                        continue
                    if re.match(r"^[A-ZÉÈÀ]", nxt) and len(nxt) > 2:
                        champs["Domicile"] = "DJIBOUTI" if re.search(djibouti_re, nxt, re.IGNORECASE) else nxt
                        break

        # Numéro de permis
        if not champs["N° permis"]:
            m = re.search(r"\b(\d{3,5}[\/\-]\d{2})\b", ligne)
            if m and not re.search(r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}", ligne):
                champs["N° permis"] = m.group(1)

    return champs


# ── Fallback regex carte grise ────────────────────────────────────────────────

def extraire_champs_carte_grise_regex(text: str) -> dict:
    """
    Extraction partielle de la carte grise par regex.
    Utilisé uniquement en fallback si Groq Vision est indisponible.
    """
    champs = {
        "immatriculation": None, "date_mise_en_circ": None, "nom": None,
        "domicile": None, "marque": None, "modele": None,
        "energie": None, "numero_serie": None, "profession": None,
    }
    for ligne in [l.strip() for l in text.split("\n") if l.strip()]:
        if not champs["immatriculation"]:
            m = re.search(r"\b([0-9]{2,4}[A-Z]{1,3}[0-9]{1,3}|[A-Z]{1,3}[0-9]{2,4}[A-Z]{0,2})\b", ligne.upper())
            if m and len(m.group(1)) <= 8:
                champs["immatriculation"] = m.group(1)
        if not champs["date_mise_en_circ"]:
            m = re.search(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b", ligne)
            if m:
                champs["date_mise_en_circ"] = f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/{m.group(3)}"
        if not champs["numero_serie"]:
            m = re.search(r"\b([A-Z0-9]{15,20})\b", ligne.upper())
            if m:
                champs["numero_serie"] = m.group(1)
        if not champs["energie"]:
            for e in ["DIESEL", "ESSENCE", "ELECTRIQUE", "HYBRIDE", "GPL"]:
                if e in ligne.upper():
                    champs["energie"] = e
                    break
    return champs
