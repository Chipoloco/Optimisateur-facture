"""
Optimisateur de Puissance Souscrite TURPE 7
Interface Streamlit — Enedis uniquement
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from turpe_engine import (
    charger_fichier_enedis,
    resumer_chargement,
    classifier_dataframe,
    calculer_cout_total,
    optimiser_puissances,
    PLAGES_HTA, PLAGES_BT_SUP,
    HTA_BI, BT_SUP_BI,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Optimisateur TURPE 7", page_icon="⚡", layout="wide")

st.title("⚡ Optimisateur de Puissance Souscrite")
st.caption("TURPE 7 — Enedis | Délibération CRE n°2025-78 | En vigueur au 1er août 2025")
st.divider()

COULEURS_PLAGES = {
    "Pointe": "#FF4444", "HPH": "#FF8C00", "HCH": "#FFD700",
    "HPB": "#4CAF50",    "HCB": "#2196F3", "unique": "#9C27B0",
}

# ─────────────────────────────────────────────
# SIDEBAR — PARAMÉTRAGE
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Paramétrage")

    domaine = st.selectbox(
        "Domaine de tension",
        ["HTA", "BT > 36 kVA", "BT ≤ 36 kVA"],
        help="Niveau de raccordement au réseau Enedis"
    )

    if domaine == "HTA":
        fta_options = ["CU pointe fixe", "CU pointe mobile", "LU pointe fixe", "LU pointe mobile"]
    elif domaine == "BT > 36 kVA":
        fta_options = ["CU", "LU"]
    else:
        fta_options = ["CU4", "MU4", "LU", "CU (dérogatoire)", "MUDT (dérogatoire)"]

    fta = st.selectbox("Formule Tarifaire d'Acheminement (FTA)", fta_options)

    type_contrat = st.selectbox(
        "Type de contrat",
        ["contrat_unique", "CARD"],
        format_func=lambda x: "Contrat unique (via fournisseur)" if x == "contrat_unique" else "CARD (contrat direct Enedis)",
    )

    st.divider()
    st.subheader("📋 Puissances actuelles (kVA)")
    st.caption("Puissances souscrites dans votre contrat actuel")

    plages = PLAGES_HTA if domaine == "HTA" else PLAGES_BT_SUP if domaine == "BT > 36 kVA" else ["unique"]

    ps_actuelles = {}
    for plage in plages:
        ps_actuelles[plage] = st.number_input(
            f"PS {plage} (kVA)", min_value=1, max_value=10000, value=100, step=1, key=f"ps_{plage}"
        )

    st.divider()
    st.subheader("⚙️ Optimisation")
    pas_kva = st.slider("Pas de balayage (kVA)", 1, 10, 1,
                        help="Précision de l'optimisation — 1 kVA recommandé")


# ─────────────────────────────────────────────
# IMPORT DES DONNÉES
# ─────────────────────────────────────────────
st.header("📂 Import de la courbe de charge")

col_import, col_format = st.columns([2, 1])

with col_import:
    uploaded_file = st.file_uploader(
        "Importez votre export Enedis SGE (format R63, CSV)",
        type=["csv"],
        help="Fichier CSV Enedis avec colonnes : Horodate, Valeur, Unité (W), Pas (PT5M)"
    )

with col_format:
    st.info("""
    **Format Enedis R63 attendu**

    Séparateur : **;**  
    Unité : **W** (Watts)  
    Pas : **PT5M** (5 min)  

    Colonnes utilisées :
    - `Horodate` → timestamp
    - `Valeur` → puissance (÷1000 → kW)
    - `Identifiant PRM` → numéro compteur

    Le programme s'adapte automatiquement
    à **n'importe quelle durée** de fichier.
    """)

# Chargement
if uploaded_file:
    try:
        df_raw = charger_fichier_enedis(uploaded_file)
        st.session_state["df_raw"] = df_raw
        resume = resumer_chargement(df_raw)

        # Bandeau de résumé du chargement
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📅 Durée", f"{resume['Nombre de jours']} jours")
        c2.metric("📊 Couverture", resume["Couverture annuelle"])
        c3.metric("🔢 Points", f"{resume['Points horaires']:,}")
        c4.metric("⚡ Pmax", f"{resume['Puissance max (kW)']} kW")
        c5.metric("🔄 Annualisation", resume["Facteur d'annualisation"])

        if df_raw.attrs.get("nb_jours", 365) < 90:
            st.warning("⚠️ Moins de 3 mois de données — les résultats sont extrapolés sur l'année entière, à interpréter avec prudence.")
        elif df_raw.attrs.get("nb_jours", 365) < 365:
            st.info(f"ℹ️ {resume['Nombre de jours']} jours de données disponibles. Les coûts sont automatiquement extrapolés sur 12 mois.")
        else:
            st.success(f"✅ {resume['Nombre de jours']} jours de données chargés — couverture optimale pour l'analyse.")

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement : {e}")
        st.stop()

else:
    # Données de démo si rien n'est chargé
    if st.button("🎲 Générer des données de démonstration (90 jours)", type="secondary"):
        np.random.seed(42)
        # Simulation 90 jours de données à 5 min
        dates   = pd.date_range("2025-01-01", periods=90 * 24 * 12, freq="5min")
        base    = 20 + 10 * np.sin(np.linspace(0, 8 * np.pi, len(dates)))
        bruit   = np.random.normal(0, 3, len(dates))
        pics    = np.zeros(len(dates))
        pics[np.random.choice(len(dates), 30, replace=False)] = np.random.uniform(15, 35, 30)
        valeurs = np.clip(base + bruit + pics, 0, 55) * 1000  # en Watts

        csv_demo = pd.DataFrame({
            "Identifiant PRM": "30001234567890",
            "Date de début": "2025-01-01 00:00:00",
            "Date de fin": "2025-04-01 00:00:00",
            "Grandeur physique": "PA",
            "Grandeur métier": "CONS",
            "Etape métier": "BEST",
            "Unité": "W",
            "Horodate": dates,
            "Valeur": valeurs.astype(int),
            "Nature": "R",
            "Pas": "PT5M",
            "Indice de vraisemblance": "null",
            "Etat complémentaire": "null",
        }).to_csv(sep=";", index=False)

        df_raw = charger_fichier_enedis(io.StringIO(csv_demo))
        st.session_state["df_raw"] = df_raw
        st.success("✅ Données de démonstration générées (90 jours, format R63)")
        st.rerun()

    if "df_raw" not in st.session_state:
        st.info("👆 Importez un fichier Enedis R63 ou générez des données de démonstration pour commencer.")
        st.stop()


# ─────────────────────────────────────────────
# ANALYSE ET OPTIMISATION
# ─────────────────────────────────────────────
df_raw = st.session_state["df_raw"]
df     = classifier_dataframe(df_raw, domaine, fta)

st.divider()
st.header("📊 Analyse de la courbe de charge")

# ── Courbe de charge colorée par plage ────────────────────────────────────────
fig_courbe = go.Figure()
for plage in sorted(df["plage"].unique()):
    df_p = df[df["plage"] == plage]
    fig_courbe.add_trace(go.Scatter(
        x=df_p["timestamp"], y=df_p["puissance_kw"],
        mode="markers",
        marker=dict(size=2, color=COULEURS_PLAGES.get(plage, "#888")),
        name=plage,
    ))

fig_courbe.update_layout(
    title="Courbe de charge par plage horosaisonnière",
    xaxis_title="Date", yaxis_title="Puissance (kW)",
    height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_courbe, use_container_width=True)

# ── Stats par plage ────────────────────────────────────────────────────────────
col_stat, col_info = st.columns([3, 1])
with col_stat:
    stats = df.groupby("plage")["puissance_kw"].agg(
        Heures="count",
        Moy=lambda x: round(x.mean(), 1),
        P90=lambda x: round(x.quantile(0.90), 1),
        P95=lambda x: round(x.quantile(0.95), 1),
        Max=lambda x: round(x.max(), 1),
    ).rename(columns={"Moy": "Moy (kW)", "P90": "P90 (kW)", "P95": "P95 (kW)", "Max": "Max (kW)"})
    stats.index.name = "Plage"
    st.dataframe(stats, use_container_width=True)

with col_info:
    fact = df_raw.attrs.get("facteur_annualisation", 1.0)
    nb_j = df_raw.attrs.get("nb_jours", 365)
    st.info(f"""
    **Annualisation**

    Durée fichier : **{nb_j} jours**

    Facteur appliqué : **×{fact}**

    Parts annualisées :
    - ✅ CS énergie (ci × Ei)
    - ✅ CMDPS

    Parts non annualisées :
    - 🔒 CG, CC (fixes/an)
    - 🔒 CS puissance (bi × P)
    """)

st.divider()

# ─────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────
st.header("💡 Optimisation des puissances souscrites")

with st.spinner("⏳ Optimisation en cours..."):
    resultat_actuel  = calculer_cout_total(df.copy(), domaine, fta, ps_actuelles, type_contrat)
    resultat_optimal, df_scenarios = optimiser_puissances(df.copy(), domaine, fta, type_contrat, pas_kva)

economie     = resultat_actuel["Total"] - resultat_optimal["Total"]
economie_pct = (economie / resultat_actuel["Total"] * 100) if resultat_actuel["Total"] > 0 else 0

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("💰 Coût TURPE actuel (€/an)",   f"{resultat_actuel['Total']:,.0f} €")
c2.metric("✅ Coût TURPE optimisé (€/an)",  f"{resultat_optimal['Total']:,.0f} €",
          delta=f"-{economie:,.0f} €")
c3.metric("📉 Économie annuelle potentielle", f"{economie:,.0f} €",
          delta=f"{economie_pct:.1f} %")

st.divider()

# ── Tableau comparatif PS ──────────────────────────────────────────────────────
st.subheader("📋 Puissances souscrites : actuel vs optimal")

df_comp = pd.DataFrame({
    "Plage":             list(ps_actuelles.keys()),
    "PS actuelle (kVA)": [ps_actuelles[p] for p in ps_actuelles],
    "PS optimisée (kVA)":[resultat_optimal["puissances_souscrites"].get(p, 0) for p in ps_actuelles],
})
df_comp["Écart (kVA)"] = df_comp["PS optimisée (kVA)"] - df_comp["PS actuelle (kVA)"]

def style_ecart(v):
    if v < 0:  return "color: green; font-weight: bold"
    if v > 0:  return "color: red"
    return ""

st.dataframe(
    df_comp.style.applymap(style_ecart, subset=["Écart (kVA)"]),
    use_container_width=True, hide_index=True
)

# ── Détail composantes ─────────────────────────────────────────────────────────
st.subheader("🔍 Détail des composantes TURPE (€/an annualisés)")

composantes = ["CG", "CC", "CS", "CMDPS"]
df_compo = pd.DataFrame({
    "Composante":       composantes,
    "Actuel (€/an)":   [resultat_actuel[c]  for c in composantes],
    "Optimisé (€/an)": [resultat_optimal[c] for c in composantes],
})
df_compo["Écart (€/an)"] = df_compo["Optimisé (€/an)"] - df_compo["Actuel (€/an)"]

col_tab, col_chart = st.columns(2)
with col_tab:
    st.dataframe(df_compo, use_container_width=True, hide_index=True)

with col_chart:
    fig_bar = go.Figure(data=[
        go.Bar(name="Actuel",   x=composantes,
               y=[resultat_actuel[c]  for c in composantes], marker_color="#FF6B6B"),
        go.Bar(name="Optimisé", x=composantes,
               y=[resultat_optimal[c] for c in composantes], marker_color="#4CAF50"),
    ])
    fig_bar.update_layout(
        barmode="group", title="Composantes : actuel vs optimisé",
        yaxis_title="€/an", height=280,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Courbe de charge avec seuils optimaux ─────────────────────────────────────
st.subheader("📉 Courbe de charge et puissances souscrites optimisées")

fig_final = go.Figure()
fig_final.add_trace(go.Scatter(
    x=df["timestamp"], y=df["puissance_kw"],
    mode="lines", name="Consommation",
    line=dict(color="#2196F3", width=1),
))
for plage, ps in resultat_optimal["puissances_souscrites"].items():
    fig_final.add_hline(
        y=ps,
        line_dash="dot",
        line_color=COULEURS_PLAGES.get(plage, "#888"),
        annotation_text=f"PS {plage} : {ps} kVA",
        annotation_position="right",
    )
fig_final.update_layout(
    title="Courbe de charge et seuils de puissances souscrites optimisées",
    xaxis_title="Date", yaxis_title="kW / kVA",
    height=380,
)
st.plotly_chart(fig_final, use_container_width=True)

# ── Sensibilité ────────────────────────────────────────────────────────────────
if "plage_variee" in df_scenarios.columns:
    st.subheader("📈 Analyse de sensibilité")
    plage_sel = st.selectbox("Plage à analyser", df_scenarios["plage_variee"].unique())
    df_sens   = df_scenarios[df_scenarios["plage_variee"] == plage_sel]

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=df_sens["ps_variee"], y=df_sens["Total"],
        mode="lines+markers", name="Coût total (€/an)",
        line=dict(color="#2196F3", width=2),
    ))
    fig_sens.add_vline(
        x=resultat_optimal["puissances_souscrites"].get(plage_sel, 0),
        line_dash="dash", line_color="#4CAF50", annotation_text="✅ Optimal",
    )
    fig_sens.add_vline(
        x=ps_actuelles.get(plage_sel, 0),
        line_dash="dash", line_color="#FF4444", annotation_text="📌 Actuel",
    )
    fig_sens.update_layout(
        title=f"Sensibilité du coût annuel à la puissance souscrite — {plage_sel}",
        xaxis_title="Puissance souscrite (kVA)",
        yaxis_title="Coût TURPE annualisé (€/an)",
        height=340,
    )
    st.plotly_chart(fig_sens, use_container_width=True)

# ── Export ─────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("💾 Export des résultats")

from datetime import datetime
rapport = pd.DataFrame({
    "Paramètre": [
        "Domaine", "FTA", "Contrat",
        "PRM", "Période", "Durée (jours)", "Couverture", "Facteur annualisation",
        "Coût actuel (€/an)", "Coût optimisé (€/an)", "Économie (€/an)", "Économie (%)",
    ],
    "Valeur": [
        domaine, fta, type_contrat,
        df_raw.attrs.get("prm", "?"),
        f"{df_raw.attrs.get('periode_debut', '?')} → {df_raw.attrs.get('periode_fin', '?')}",
        df_raw.attrs.get("nb_jours", "?"),
        f"{round(df_raw.attrs.get('nb_jours', 365) / 365 * 100, 1)} %",
        f"×{df_raw.attrs.get('facteur_annualisation', 1.0)}",
        f"{resultat_actuel['Total']:,.0f} €",
        f"{resultat_optimal['Total']:,.0f} €",
        f"{economie:,.0f} €",
        f"{economie_pct:.1f} %",
    ]
})

# Ajout des PS par plage
for plage in ps_actuelles:
    rapport = pd.concat([rapport, pd.DataFrame({
        "Paramètre": [f"PS actuelle {plage}", f"PS optimisée {plage}"],
        "Valeur": [
            f"{ps_actuelles[plage]} kVA",
            f"{resultat_optimal['puissances_souscrites'].get(plage, '?')} kVA",
        ]
    })], ignore_index=True)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "📥 Rapport de synthèse (CSV)",
        data=rapport.to_csv(index=False, sep=";"),
        file_name=f"rapport_turpe7_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
with col_dl2:
    st.download_button(
        "📥 Courbe classée avec plages (CSV)",
        data=df[["timestamp", "puissance_kw", "plage"]].to_csv(index=False, sep=";"),
        file_name=f"courbe_classee_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("📌 TURPE 7 en vigueur au 1er août 2025 — CRE n°2025-78 — Périmètre Enedis uniquement — Coûts affichés en €/an annualisés")
