"""
scanner.py
----------
Scanner virtuel universel : détecte les bords du document
dans une photo, corrige la perspective et prétraite l'image.
Fonctionne pour le permis ET le certificat d'immatriculation.
"""

import cv2
import numpy as np
import imutils
import os

from pretraitement import pretraiter_image


def order_points(pts):
    """Réordonne 4 points : haut-gauche, haut-droit, bas-droit, bas-gauche."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """
    Transformation perspective sur l'image ORIGINALE haute résolution.
    La détection des coins se fait sur image réduite (500px) pour la vitesse,
    mais le redressement final utilise chaque pixel disponible.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA   = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB   = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA   = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB   = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def scanner_document(image_path: str, tmp_dir: str = ".") -> tuple:
    """
    Scanner virtuel universel.

    Paramètres
    ----------
    image_path : chemin vers la photo brute
    tmp_dir    : dossier où écrire les fichiers intermédiaires

    Retourne
    --------
    (warped_color, warped_thresh, warped_path)
    warped_path : chemin du fichier image redressée (couleur)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    orig  = image.copy()
    ratio = image.shape[0] / 500.0
    small = imutils.resize(image, height=500)

    # ── Masque HSV : isole les zones claires ─────────────────────────────────
    hsv  = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 80, 255]))
    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)

    # ── Détection des bords ───────────────────────────────────────────────────
    gray    = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    masked  = cv2.bitwise_and(gray, gray, mask=mask)
    blurred = cv2.GaussianBlur(masked, (21, 21), 0)
    edged   = cv2.Canny(blurred, 10, 50)
    k2      = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edged   = cv2.dilate(edged, k2, iterations=2)
    edged   = cv2.erode(edged,  k2, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    # ── Cas : aucun contour (image trop sombre ou uniforme) ───────────────────
    if not cnts:
        print("  [scanner] Aucun contour détecté — image utilisée en entier")
        warped = orig.copy()
        _, warped_thresh = pretraiter_image(warped)
        warped_path = os.path.join(tmp_dir, "doc_warped.jpg")
        cv2.imwrite(warped_path, warped)
        return warped, warped_thresh, warped_path

    # ── Recherche du contour à 4 coins ───────────────────────────────────────
    image_area = small.shape[0] * small.shape[1]
    screen_cnt = None

    for c in cnts:
        if cv2.contourArea(c) < 0.15 * image_area:
            continue
        peri = cv2.arcLength(c, True)
        for eps in [0.02, 0.03, 0.05, 0.08, 0.10]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                screen_cnt = approx
                break
        if screen_cnt is not None:
            break

    if screen_cnt is None:
        print("  [scanner] Fallback -> bounding rectangle")
        x, y, w, h = cv2.boundingRect(cnts[0])
        screen_cnt = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])

    # ── Transformation perspective ────────────────────────────────────────────
    warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)
    _, warped_thresh = pretraiter_image(warped)

    warped_path = os.path.join(tmp_dir, "doc_warped.jpg")
    cv2.imwrite(warped_path, warped)

    return warped, warped_thresh, warped_path
