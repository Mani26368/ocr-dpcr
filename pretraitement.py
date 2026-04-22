"""
pretraitement.py
----------------
Fonctions de prétraitement d'image appliquées avant le scanner
et avant l'envoi à Groq Vision.
"""

import cv2
import numpy as np
from skimage.filters import threshold_local


# ── Prétraitement N&B (pour OCR fallback) ────────────────────────────────────

def ameliorer_contraste(image_gray):
    """
    CLAHE : correction locale du contraste zone par zone.
    Indispensable pour le permis djiboutien (fond rose pastel)
    et la carte grise (fond blanc avec zones sombres).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_gray)


def reduire_bruit(image_gray):
    """
    Non-Local Means Denoising : supprime le bruit de capteur
    sans flouter les bords des lettres.
    Critique pour les photos WhatsApp très compressées.
    """
    return cv2.fastNlMeansDenoising(
        image_gray, h=10, templateWindowSize=7, searchWindowSize=21
    )


def binariser_adaptatif(image_gray):
    """
    Seuillage adaptatif de Sauvola : calcule un seuil différent
    pour chaque fenêtre de 11x11 pixels.
    Gère l'éclairage inégal des photos de terrain.
    """
    T = threshold_local(image_gray, block_size=11, offset=10, method="gaussian")
    return (image_gray > T).astype("uint8") * 255


def pretraiter_image(image_bgr):
    """Pipeline complet N&B → retourne (gray, binary)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = ameliorer_contraste(gray)
    gray = reduire_bruit(gray)
    binary = binariser_adaptatif(gray)
    return gray, binary


# ── Normalisation avant envoi à Groq Vision ──────────────────────────────────

def normaliser_pour_groq(img_bgr):
    """
    Normalise l'image avant envoi à Groq Vision.
    Corrige les cas surexposés (fond trop blanc) et sous-exposés (trop sombres)
    qui font halluciner le modèle sur les champs textuels.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()

    if mean_val > 210:
        img_bgr = np.clip(img_bgr.astype(np.float32) * 0.65, 0, 255).astype(np.uint8)
        print(f"  [pretraitement] Surexposé ({mean_val:.0f}) -> luminosité corrigée")
    elif mean_val < 60:
        img_bgr = np.clip(img_bgr.astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
        print(f"  [pretraitement] Sous-exposé ({mean_val:.0f}) -> luminosité corrigée")

    # Débruitage léger couleur
    img_bgr = cv2.fastNlMeansDenoisingColored(
        img_bgr, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21
    )
    return img_bgr
