"""
supabase_client.py
------------------
Persistance des résultats JSON dans Supabase.
Utilise upsert pour éviter les doublons.

Table 'permis'      pour les permis de conduire
Table 'carte_grise' pour les certificats d'immatriculation
"""

import requests
from config import SUPABASE_URL, SUPABASE_KEY


def sauvegarder_document(resultat: dict) -> bool:
    """
    Sauvegarde le résultat du pipeline dans Supabase.
    Retourne True si la sauvegarde a réussi, False sinon.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[supabase] Configuration manquante — sauvegarde ignorée")
        return False

    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "resolution=merge-duplicates,return=representation",
    }

    type_doc = resultat.get("type_document", "permis")

    if type_doc == "carte_grise":
        table        = "carte_grise"
        conflict_col = "immatriculation"
        payload = {
            "immatriculation":   resultat.get("immatriculation"),
            "date_mise_en_circ": resultat.get("date_mise_en_circ"),
            "nom":               resultat.get("nom"),
            "domicile":          resultat.get("domicile"),
            "marque":            resultat.get("marque"),
            "modele":            resultat.get("modele"),
            "energie":           resultat.get("energie"),
            "numero_serie":      resultat.get("numero_serie"),
            "profession":        resultat.get("profession"),
        }
    else:
        table        = "permis"
        conflict_col = "numero_permis"
        payload = {
            "nom":            resultat.get("nom"),
            "date_naissance": resultat.get("date_naissance"),
            "lieu_naissance": resultat.get("lieu_naissance"),
            "domicile":       resultat.get("domicile"),
            "numero_permis":  resultat.get("numero_permis"),
            "categorie":      resultat.get("categorie"),
        }

    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={conflict_col}",
            headers=headers,
            json=payload,
            timeout=10,
        )
        if resp.status_code in (200, 201):
            print(f"[supabase] Sauvegardé dans '{table}'")
            return True
        elif resp.status_code == 409:
            print(f"[supabase] Document existant mis à jour dans '{table}'")
            return True
        else:
            print(f"[supabase] Erreur {resp.status_code} : {resp.text}")
            return False
    except Exception as e:
        print(f"[supabase] Exception : {e}")
        return False
