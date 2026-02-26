"""
Moteur de calcul TURPE 7 — Enedis
En vigueur au 1er août 2025 (Délibération CRE n°2025-78 du 13 mars 2025)

Périmètre :
- HTA       : 5 plages (Pointe, HPH, HCH, HPB, HCB), 4 FTA
- BT > 36   : 4 plages (HPH, HCH, HPB, HCB), 2 FTA
- BT ≤ 36   : puissance unique, 5 FTA (pas de CMDPS)

Annualisation automatique :
- Tous les coûts sont exprimés en €/AN quelle que soit la durée du fichier.
- CG, CC                : fixes annuels → pas de correction.
- CS part puissance (bi × P) : abonnement annuel → pas de correction.
- CS part énergie (ci × Ei)  : proportionnelle à l'énergie → × (365 / nb_jours).
- CMDPS                      : calculée sur les mois présents → × (365 / nb_jours).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────────────────────
# 1. CONSTANTES TARIFAIRES TURPE 7
# ─────────────────────────────────────────────

PLAGES_HTA    = ["Pointe", "HPH", "HCH", "HPB", "HCB"]
PLAGES_BT_SUP = ["HPH", "HCH", "HPB", "HCB"]

# HTA : bi (€/kW/an)
HTA_BI = {
    "CU pointe fixe":   {"Pointe": 14.41, "HPH": 14.41, "HCH": 14.41, "HPB": 12.55, "HCB": 11.22},
    "CU pointe mobile": {"Pointe": 14.41, "HPH": 14.41, "HCH": 14.41, "HPB": 12.55, "HCB": 11.22},
    "LU pointe fixe":   {"Pointe": 35.33, "HPH": 32.30, "HCH": 20.39, "HPB": 14.33, "HCB": 11.56},
    "LU pointe mobile": {"Pointe": 38.27, "HPH": 34.30, "HCH": 20.39, "HPB": 14.33, "HCB": 11.56},
}

# HTA : ci (c€/kWh)
HTA_CI = {
    "CU pointe fixe":   {"Pointe": 5.74, "HPH": 4.23, "HCH": 1.99, "HPB": 1.01, "HCB": 0.69},
    "CU pointe mobile": {"Pointe": 7.01, "HPH": 4.05, "HCH": 1.99, "HPB": 1.01, "HCB": 0.69},
    "LU pointe fixe":   {"Pointe": 2.65, "HPH": 2.10, "HCH": 1.47, "HPB": 0.92, "HCB": 0.68},
    "LU pointe mobile": {"Pointe": 3.15, "HPH": 1.87, "HCH": 1.47, "HPB": 0.92, "HCB": 0.68},
}

# BT > 36 kVA : bi (€/kVA/an)
BT_SUP_BI = {
    "CU": {"HPH": 17.61, "HCH": 15.96, "HPB": 14.56, "HCB": 11.98},
    "LU": {"HPH": 30.16, "HCH": 21.18, "HPB": 16.64, "HCB": 12.37},
}

# BT > 36 kVA : ci (c€/kWh)
BT_SUP_CI = {
    "CU": {"HPH": 6.91, "HCH": 4.21, "HPB": 2.13, "HCB": 1.52},
    "LU": {"HPH": 5.69, "HCH": 3.47, "HPB": 2.01, "HCB": 1.49},
}

# BT ≤ 36 kVA : b unique (€/kVA/an)
BT_INF_B = {
    "CU4":                10.11,
    "MU4":                12.12,
    "LU":                 93.13,
    "CU (dérogatoire)":   11.07,
    "MUDT (dérogatoire)": 13.49,
}

# BT ≤ 36 kVA : ci (c€/kWh)
BT_INF_CI = {
    "CU4":  {"HPH": 7.49, "HCH": 3.97, "HPB": 1.66, "HCB": 1.16},
    "MU4":  {"HPH": 7.00, "HCH": 3.73, "HPB": 1.61, "HCB": 1.11},
    "LU":   {"HPH": 1.25, "HCH": 1.25, "HPB": 1.25, "HCB": 1.25},
    "CU (dérogatoire)":   {"HPH": 4.84, "HCH": 4.84, "HPB": 4.84, "HCB": 4.84},
    "MUDT (dérogatoire)": {"HPH": 4.94, "HCH": 3.50, "HPB": 4.94, "HCB": 3.50},
}

# Composantes fixes (€/an)
COMPOSANTES_FIXES = {
    "HTA":    {"CG_contrat_unique": 435.72, "CG_CARD": 499.80, "CC": 376.39},
    "BT_SUP": {"CG_contrat_unique": 217.80, "CG_CARD": 249.84, "CC": 283.27},
    "BT_INF": {"CG_contrat_unique":  16.80, "CG_CARD":  18.00, "CC":  22.00},
}


# ─────────────────────────────────────────────
# 2. INGESTION DES DONNÉES ENEDIS SGE (R63)
# ─────────────────────────────────────────────

def charger_fichier_enedis(filepath_or_buffer) -> pd.DataFrame:
    """
    Charge un fichier CSV Enedis SGE format R63 (courbe de charge).
    Adapte automatiquement les calculs à la période réelle du fichier.

    Retourne un DataFrame horaire avec :
    - timestamp           : datetime
    - puissance_kw        : puissance max sur l'heure (kW)
    - puissance_kw_10min  : puissance max sur 10 min (kW) — CMDPS HTA

    Attrs : prm, periode_debut, periode_fin, nb_jours, facteur_annualisation
    """
    df = pd.read_csv(filepath_or_buffer, sep=";", parse_dates=["Horodate"])

    colonnes_requises = {"Horodate", "Valeur", "Unité"}
    if not colonnes_requises.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes. Attendues : {colonnes_requises}. Trouvées : {set(df.columns)}")

    if not (df["Unité"] == "W").all():
        raise ValueError(f"Unité inattendue : {df['Unité'].unique()}. Seul 'W' est supporté.")

    # Nettoyage et conversion W → kW
    df_clean = pd.DataFrame({
        "timestamp":    df["Horodate"],
        "puissance_kw": df["Valeur"] / 1000.0,
        "prm":          df["Identifiant PRM"].iloc[0] if "Identifiant PRM" in df.columns else None,
    }).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Rééchantillonnage 10 min (max) → CMDPS HTA
    df_10min = (
        df_clean.set_index("timestamp")["puissance_kw"]
        .resample("10min").max()
        .reset_index()
        .rename(columns={"puissance_kw": "puissance_kw_10min"})
    )

    # Rééchantillonnage 1h (max) → CS et CMDPS BT
    df_1h = (
        df_clean.set_index("timestamp")["puissance_kw"]
        .resample("h").max()
        .reset_index()
    )

    # Ajout puissance max 10 min par heure
    df_1h = df_1h.merge(
        df_10min
        .assign(timestamp=df_10min["timestamp"].dt.floor("h"))
        .groupby("timestamp")["puissance_kw_10min"].max()
        .reset_index(),
        on="timestamp", how="left"
    )

    # Calcul du facteur d'annualisation
    debut    = df_clean["timestamp"].min()
    fin      = df_clean["timestamp"].max()
    nb_jours = max(1, (fin - debut).days)
    fact_ann = round(365.0 / nb_jours, 4)

    # Stockage des métadonnées dans les attrs du DataFrame
    df_1h.attrs["prm"]                    = df_clean["prm"].iloc[0]
    df_1h.attrs["periode_debut"]          = debut
    df_1h.attrs["periode_fin"]            = fin
    df_1h.attrs["nb_jours"]               = nb_jours
    df_1h.attrs["facteur_annualisation"]  = fact_ann
    df_1h.attrs["resolution_source"]      = df["Pas"].iloc[0] if "Pas" in df.columns else "inconnue"

    return df_1h


def resumer_chargement(df: pd.DataFrame) -> dict:
    """Retourne un résumé lisible du DataFrame chargé."""
    nb_jours = df.attrs.get("nb_jours", 365)
    fact     = df.attrs.get("facteur_annualisation", 1.0)
    couv_pct = round(nb_jours / 365 * 100, 1)
    return {
        "PRM":                     df.attrs.get("prm", "inconnu"),
        "Période":                 f"{df.attrs.get('periode_debut', '?')} → {df.attrs.get('periode_fin', '?')}",
        "Nombre de jours":         nb_jours,
        "Couverture annuelle":     f"{couv_pct} %",
        "Facteur d'annualisation": f"×{fact}",
        "Résolution source":       df.attrs.get("resolution_source", "?"),
        "Points horaires":         len(df),
        "Puissance min (kW)":      round(df["puissance_kw"].min(), 1),
        "Puissance moyenne (kW)":  round(df["puissance_kw"].mean(), 1),
        "Puissance max (kW)":      round(df["puissance_kw"].max(), 1),
        "Valeurs manquantes":      int(df["puissance_kw"].isna().sum()),
    }


# ─────────────────────────────────────────────
# 3. CLASSIFICATION DES HEURES
# ─────────────────────────────────────────────

def classifier_plage(timestamp: pd.Timestamp, domaine: str, fta: str) -> str:
    """
    Retourne la plage temporelle d'un timestamp.
    Saison haute : novembre à mars inclus.
    HP : 8h-20h, lundi-vendredi.
    Pointe HTA fixe : 8h-10h et 17h-19h, déc-fév, hors dimanche.
    """
    mois         = timestamp.month
    heure        = timestamp.hour
    jour_semaine = timestamp.weekday()   # 0=lundi … 6=dimanche
    saison_haute = mois in [11, 12, 1, 2, 3]

    if domaine == "HTA":
        if "pointe fixe" in fta:
            if mois in [12, 1, 2] and jour_semaine < 6:
                if 8 <= heure < 10 or 17 <= heure < 19:
                    return "Pointe"
        elif "pointe mobile" in fta:
            if mois in [12, 1, 2] and jour_semaine < 6:
                if 7 <= heure < 15 or 18 <= heure < 20:
                    return "Pointe"
        return ("HPH" if (8 <= heure < 20 and jour_semaine < 6) else "HCH") if saison_haute \
          else ("HPB" if (8 <= heure < 20 and jour_semaine < 6) else "HCB")
    else:
        return ("HPH" if (8 <= heure < 20 and jour_semaine < 6) else "HCH") if saison_haute \
          else ("HPB" if (8 <= heure < 20 and jour_semaine < 6) else "HCB")


def classifier_dataframe(df: pd.DataFrame, domaine: str, fta: str) -> pd.DataFrame:
    """Ajoute une colonne 'plage' au DataFrame en préservant ses attrs."""
    df = df.copy()
    df["plage"] = df["timestamp"].apply(lambda t: classifier_plage(t, domaine, fta))
    return df


# ─────────────────────────────────────────────
# 4. CALCUL DU COÛT TURPE ANNUALISÉ
# ─────────────────────────────────────────────

def calculer_cout_total(
    df: pd.DataFrame,
    domaine: str,
    fta: str,
    puissances_souscrites: Dict[str, float],
    type_contrat: str = "contrat_unique",
) -> Dict:
    """
    Calcule le coût TURPE annuel (€/an) pour une configuration donnée.
    S'adapte automatiquement à la durée des données via facteur_annualisation.

    Règle d'annualisation :
    - CG, CC, CS part puissance : toujours annuels, pas de correction.
    - CS part énergie + CMDPS   : × facteur_annualisation (= 365 / nb_jours).
    """
    domaine_key = {"HTA": "HTA", "BT > 36 kVA": "BT_SUP", "BT ≤ 36 kVA": "BT_INF"}[domaine]
    fixes    = COMPOSANTES_FIXES[domaine_key]
    cg       = fixes[f"CG_{type_contrat}"]
    cc       = fixes["CC"]
    fact_ann = df.attrs.get("facteur_annualisation", 1.0)

    # Énergies réelles sur la période du fichier (kWh)
    energies = df.groupby("plage")["puissance_kw"].sum().to_dict()

    if domaine == "HTA":
        bi = HTA_BI[fta]
        ci = HTA_CI[fta]
        ordre    = sorted(PLAGES_HTA, key=lambda p: bi[p])
        ps_tries = [puissances_souscrites[p] for p in ordre]

        cs_puissance = bi[ordre[0]] * ps_tries[0]
        for idx in range(1, len(ordre)):
            cs_puissance += bi[ordre[idx]] * (ps_tries[idx] - ps_tries[idx - 1])
        cs_energie = sum((ci[p] / 100) * energies.get(p, 0) * fact_ann for p in PLAGES_HTA)
        cs = cs_puissance + cs_energie

        # CMDPS HTA : utilise puissance_kw_10min si disponible
        col_p = "puissance_kw_10min" if "puissance_kw_10min" in df.columns else "puissance_kw"
        cmdps = 0.0
        for (_, plage), groupe in df.groupby([df["timestamp"].dt.month, "plage"]):
            ps     = puissances_souscrites.get(plage, 0)
            deltas = np.maximum(0, groupe[col_p].values - ps)
            if deltas.sum() > 0:
                cmdps += 0.04 * bi[plage] * np.sqrt(np.sum(deltas ** 2))
        cmdps *= fact_ann

    elif domaine == "BT > 36 kVA":
        bi = BT_SUP_BI[fta]
        ci = BT_SUP_CI[fta]
        ordre    = sorted(PLAGES_BT_SUP, key=lambda p: bi[p])
        ps_tries = [puissances_souscrites[p] for p in ordre]

        cs_puissance = bi[ordre[0]] * ps_tries[0]
        for idx in range(1, len(ordre)):
            cs_puissance += bi[ordre[idx]] * (ps_tries[idx] - ps_tries[idx - 1])
        cs_energie = sum((ci[p] / 100) * energies.get(p, 0) * fact_ann for p in PLAGES_BT_SUP)
        cs = cs_puissance + cs_energie

        # CMDPS BT > 36 : 12,41 €/h de dépassement
        heures_dep = sum(
            (df[df["plage"] == p]["puissance_kw"] > ps).sum()
            for p, ps in puissances_souscrites.items()
        )
        cmdps = 12.41 * heures_dep * fact_ann

    else:  # BT ≤ 36 kVA
        ps_unique = list(puissances_souscrites.values())[0]
        b  = BT_INF_B[fta]
        ci = BT_INF_CI[fta]
        cs_puissance = b * ps_unique
        cs_energie   = sum((ci[p] / 100) * energies.get(p, 0) * fact_ann for p in ci)
        cs    = cs_puissance + cs_energie
        cmdps = 0.0

    total = cg + cc + cs + cmdps

    return {
        "CG":                    round(cg, 2),
        "CC":                    round(cc, 2),
        "CS":                    round(cs, 2),
        "CMDPS":                 round(cmdps, 2),
        "Total":                 round(total, 2),
        "facteur_annualisation": fact_ann,
        "puissances_souscrites": puissances_souscrites,
    }


# ─────────────────────────────────────────────
# 5. MOTEUR D'OPTIMISATION
# ─────────────────────────────────────────────

def optimiser_puissances(
    df: pd.DataFrame,
    domaine: str,
    fta: str,
    type_contrat: str = "contrat_unique",
    pas_kva: int = 1,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Optimise les puissances souscrites par plage temporelle.
    Tous les coûts sont exprimés en €/an annualisés.

    Retourne :
    - Dict : PS optimales + coût annualisé détaillé
    - DataFrame : scénarios de sensibilité autour de l'optimal
    """
    plages = (
        PLAGES_HTA     if domaine == "HTA"         else
        PLAGES_BT_SUP  if domaine == "BT > 36 kVA" else
        ["unique"]
    )

    if domaine in ["HTA", "BT > 36 kVA"]:
        bi = HTA_BI[fta] if domaine == "HTA" else BT_SUP_BI[fta]

        # Puissance max/min par plage — fallback à 1 kVA si la plage est absente du fichier
        p_global_max  = df["puissance_kw"].max()
        max_par_plage = {p: (df[df["plage"] == p]["puissance_kw"].max() if (df["plage"] == p).any() else p_global_max) for p in plages}
        min_par_plage = {p: max(1, (df[df["plage"] == p]["puissance_kw"].quantile(0.50) if (df["plage"] == p).any() else 1)) for p in plages}

        candidats = {
            p: list(range(
                max(1, int(min_par_plage[p]) - pas_kva),
                int(max_par_plage[p]) + pas_kva * 2,
                pas_kva
            )) for p in plages
        }

        # Optimisation indépendante par plage
        meilleures_ps = {}
        for plage in plages:
            meilleur_cout = float("inf")
            meilleure_ps  = max_par_plage[plage]
            for ps_cand in candidats[plage]:
                ps_test = {p: max_par_plage[p] for p in plages}
                ps_test[plage] = ps_cand
                r = calculer_cout_total(df.copy(), domaine, fta, ps_test, type_contrat)
                if r["Total"] < meilleur_cout:
                    meilleur_cout = r["Total"]
                    meilleure_ps  = ps_cand
            meilleures_ps[plage] = meilleure_ps

        # Contrainte TURPE : Pi+1 >= Pi (ordre croissant des bi)
        for p in sorted(plages, key=lambda p: bi[p]):
            ps_max_cumul = max((meilleures_ps[q] for q in plages if bi[q] < bi[p]), default=0)
            meilleures_ps[p] = max(meilleures_ps[p], ps_max_cumul)

        resultat_optimal = calculer_cout_total(df.copy(), domaine, fta, meilleures_ps, type_contrat)

        # Scénarios de sensibilité
        scenarios = []
        for plage in plages:
            ps_opt = meilleures_ps[plage]
            for delta in range(-5 * pas_kva, 6 * pas_kva, pas_kva):
                ps_var = dict(meilleures_ps)
                ps_var[plage] = max(1, ps_opt + delta)
                r = calculer_cout_total(df.copy(), domaine, fta, ps_var, type_contrat)
                r["plage_variee"] = plage
                r["ps_variee"]    = ps_var[plage]
                scenarios.append(r)
        df_scenarios = pd.DataFrame(scenarios)

    else:  # BT ≤ 36 kVA
        p_max         = df["puissance_kw"].max()
        p_min         = max(1, df["puissance_kw"].quantile(0.50))
        scenarios     = []
        meilleur_cout = float("inf")
        meilleure_ps  = p_max

        for ps in range(max(1, int(p_min) - pas_kva), int(p_max) + pas_kva * 2, pas_kva):
            r = calculer_cout_total(df.copy(), domaine, fta, {"unique": ps}, type_contrat)
            r["ps_variee"] = ps
            scenarios.append(r)
            if r["Total"] < meilleur_cout:
                meilleur_cout = r["Total"]
                meilleure_ps  = ps

        meilleures_ps    = {"unique": meilleure_ps}
        resultat_optimal = calculer_cout_total(df.copy(), domaine, fta, meilleures_ps, type_contrat)
        df_scenarios     = pd.DataFrame(scenarios)

    return resultat_optimal, df_scenarios
