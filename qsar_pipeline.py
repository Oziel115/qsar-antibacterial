"""
QSAR Pipeline: Predicción de Actividad Antibacteriana
=====================================================
Autores: Oziel A. Carrión, Abimael Guzmán
Universidad Autónoma de Chihuahua
Facultad de Medicina y Ciencias Biomédicas

Descripción:
    Pipeline completo de QSAR para predicción de actividad antibacteriana
    en cinco patógenos de prioridad crítica (WHO 2024) usando datos de ChEMBL.
    Implementa Random Forest y XGBoost con Morgan fingerprints (ECFP4),
    scaffold split para evaluación rigurosa, y optimización con Optuna.

Patógenos:
    - Staphylococcus aureus     (CHEMBL352)
    - Escherichia coli          (CHEMBL354)
    - Pseudomonas aeruginosa    (CHEMBL348)
    - Klebsiella pneumoniae     (CHEMBL350)
    - Acinetobacter baumannii   (CHEMBL614425)

Referencias:
    [1] Ebejer et al. J Cheminform. 2016;8:30. DOI: 10.1186/s13321-016-0143-5
    [2] Rogers & Hahn. J Chem Inf Model. 2010;50(5):742-754. DOI: 10.1021/ci100050t
    [3] Li et al. Aging. 2022;14(3):1448-1465. DOI: 10.18632/aging.203887
"""

# 1. IMPORTS


import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, f1_score,
                             matthews_corrcoef, make_scorer)

from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 2. CONFIGURACIÓN


# Bacterias y sus IDs en ChEMBL
BACTERIAS = {
    "Staphylococcus_aureus":     "CHEMBL352",
    "Escherichia_coli":          "CHEMBL354",
    "Pseudomonas_aeruginosa":    "CHEMBL348",
    "Klebsiella_pneumoniae":     "CHEMBL350",
    "Acinetobacter_baumannii":   "CHEMBL614425",
}

# Umbral de clasificación (Ebejer et al. 2016)
MIC_THRESHOLD = 8.0  # µg/mL

# Parámetros de Morgan fingerprints (ECFP4, Rogers & Hahn 2010)
FP_RADIUS = 2
FP_BITS = 2048

# División de datos
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Validación cruzada
N_FOLDS = 5

# Optuna
N_TRIALS = 50

# Rutas
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
DATA_FP = "data/processed/fingerprints"
FIGURES_DIR = "figures"

os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(DATA_FP, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# 3. DESCARGA DE DATOS DE CHEMBL

def descargar_datos_chembl(target_id, nombre, max_records=None):
    """
    Descarga datos de MIC de ChEMBL para un target dado.

    Parámetros:
        target_id (str): ID del target en ChEMBL (ej. 'CHEMBL352')
        nombre (str): Nombre de la bacteria para guardar el archivo
        max_records (int): Límite de registros (None = todos)

    Retorna:
        pd.DataFrame: Datos crudos de ChEMBL
    """
    from chembl_webresource_client.new_client import new_client

    output_path = os.path.join(DATA_RAW, f"{nombre}_raw.csv")

    if os.path.exists(output_path):
        print(f"  {nombre}: cargando desde disco...")
        return pd.read_csv(output_path)

    print(f"  {nombre}: descargando de ChEMBL ({target_id})...")
    activity = new_client.activity
    filtros = activity.filter(
        target_chembl_id=target_id,
        standard_type="MIC",
        standard_relation__in=["=", "<", "<="],
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "assay_chembl_id",
    ])

    if max_records:
        filtros = filtros[:max_records]

    df = pd.DataFrame.from_dict(filtros)
    df.to_csv(output_path, index=False)
    print(f"    {len(df)} registros descargados")
    return df


# 4. PREPROCESAMIENTO

def convertir_a_ugml(row):
    """
    Convierte MIC de nM a µg/mL usando el peso molecular calculado con RDKit.
    Fórmula: µg/mL = nM × PM / 1,000,000
    """
    if row["standard_units"] == "nM":
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is None:
            return None
        mw = Descriptors.MolWt(mol)
        return row["standard_value"] * mw / 1_000_000
    return row["standard_value"]


def limpiar_dataset(df, nombre):
    """
    Aplica pipeline de limpieza completo:
    1. Elimina registros sin SMILES
    2. Elimina MIC <= 0
    3. Convierte unidades a µg/mL
    4. Elimina SMILES inválidos
    5. Resuelve duplicados con mediana
    6. Clasifica activo/inactivo con umbral de 8 µg/mL

    Parámetros:
        df (pd.DataFrame): Datos crudos de ChEMBL
        nombre (str): Nombre de la bacteria

    Retorna:
        pd.DataFrame: Dataset limpio
    """
    print(f"\n  Limpiando {nombre}:")
    print(f"    Registros iniciales: {len(df)}")

    # Paso 1: eliminar sin SMILES
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    print(f"    Tras eliminar sin SMILES: {len(df)}")

    # Paso 2: eliminar MIC <= 0
    df = df[df["standard_value"] > 0]

    # Paso 3: solo unidades relevantes
    df = df[df["standard_units"].isin(["ug/mL", "µg/mL", "nM"])]

    # Paso 4: convertir a µg/mL
    df = df.copy()
    df["mic_ugml"] = df.apply(convertir_a_ugml, axis=1)
    df = df.dropna(subset=["mic_ugml"])

    # Paso 5: eliminar SMILES inválidos
    df = df[df["canonical_smiles"].apply(
        lambda s: Chem.MolFromSmiles(s) is not None
    )]

    # Paso 6: mediana por compuesto
    df = df.groupby("molecule_chembl_id").agg(
        canonical_smiles=("canonical_smiles", "first"),
        mic_ugml=("mic_ugml", "median"),
        num_mediciones=("mic_ugml", "count"),
    ).reset_index()
    print(f"    Compuestos únicos: {len(df)}")

    # Paso 7: clasificar activo/inactivo
    df["activo"] = (df["mic_ugml"] <= MIC_THRESHOLD).astype(int)
    print(f"    Activos ({MIC_THRESHOLD} µg/mL): {df['activo'].sum()} "
          f"({df['activo'].mean()*100:.1f}%)")

    return df


# 5. SCAFFOLD SPLIT

def scaffold_split(df, test_size=TEST_SIZE):
    """
    Divide el dataset por scaffold de Murcko.
    Los scaffolds más representativos van al train set.
    Garantiza que el test contiene scaffolds no vistos durante entrenamiento.

    Parámetros:
        df (pd.DataFrame): Dataset limpio con columna 'canonical_smiles'
        test_size (float): Fracción para test (default 0.2)

    Retorna:
        tuple: (train_df, test_df)
    """
    # Calcular scaffold de cada molécula
    scaffolds = {}
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
            scaffolds.setdefault(scaffold, []).append(idx)

    # Ordenar de mayor a menor (más representativos al train)
    scaffold_sets = sorted(
        scaffolds.values(), key=lambda x: len(x), reverse=True
    )

    # Llenar train primero
    n_train = int(len(df) * (1 - test_size))
    train_idx, test_idx = [], []

    for scaffold_set in scaffold_sets:
        if len(train_idx) < n_train:
            train_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    return train_df, test_df


# 6. MORGAN FINGERPRINTS (ECFP4)

def calcular_fingerprints(df):
    """
    Calcula Morgan fingerprints (ECFP4) para cada molécula.
    Radio = 2, nBits = 2048 (Rogers & Hahn 2010).

    Parámetros:
        df (pd.DataFrame): Dataset con columna 'canonical_smiles'

    Retorna:
        tuple: (X, y) donde X es matriz de fingerprints e y son etiquetas
    """
    generador = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS, fpSize=FP_BITS
    )

    fingerprints, indices_validos = [], []

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is not None:
            fp = generador.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp)
            indices_validos.append(idx)

    X = np.array(fingerprints)
    y = df.loc[indices_validos, "activo"].values

    return X, y


# 7. VALIDACIÓN CRUZADA

def evaluar_con_cv(modelo, X, y, n_splits=N_FOLDS):
    """
    Evalúa un modelo con validación cruzada estratificada.

    Parámetros:
        modelo: Modelo de scikit-learn/XGBoost
        X (np.ndarray): Matriz de features
        y (np.ndarray): Etiquetas
        n_splits (int): Número de folds

    Retorna:
        dict: Métricas promedio (roc_auc, roc_auc_std, f1, mcc)
    """
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "mcc": make_scorer(matthews_corrcoef),
    }

    scores = cross_validate(modelo, X, y, cv=cv, scoring=scoring)

    return {
        "roc_auc":     scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(),
        "f1":          scores["test_f1"].mean(),
        "mcc":         scores["test_mcc"].mean(),
    }


# 8. EVALUACIÓN EN TEST EXTERNO


def evaluar_en_test(modelo, X_train, y_train, X_test, y_test):
    """
    Entrena el modelo con todo el train set y evalúa en test externo.
    El test externo se usa UNA SOLA VEZ para reportar el rendimiento real.

    Parámetros:
        modelo: Modelo de scikit-learn/XGBoost
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de test externo

    Retorna:
        dict: Métricas en test externo (roc_auc, f1, mcc)
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1":      f1_score(y_test, y_pred),
        "mcc":     matthews_corrcoef(y_test, y_pred),
    }


# 9. TUNING CON OPTUNA


def objetivo_xgb(trial, X, y):
    """Función objetivo para Optuna: evalúa XGBoost con CV."""
    cv = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "eval_metric":       "logloss",
        "verbosity":         0,
    }

    modelo = XGBClassifier(**params)
    scores = cross_validate(modelo, X, y, cv=cv, scoring="roc_auc")
    return scores["test_score"].mean()


def tunear_xgboost(X, y, n_trials=N_TRIALS):
    """
    Optimiza hiperparámetros de XGBoost con Optuna.
    Usa optimización bayesiana (TPE sampler).

    Parámetros:
        X (np.ndarray): Matriz de features del train set
        y (np.ndarray): Etiquetas del train set
        n_trials (int): Número de trials de Optuna

    Retorna:
        dict: Mejores hiperparámetros encontrados
    """
    estudio = optuna.create_study(direction="maximize")
    estudio.optimize(
        lambda trial: objetivo_xgb(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return estudio.best_params, estudio.best_value


# 10. VISUALIZACIONES

def plot_cv_vs_test(resultados_cv, resultados_test, nombre_modelo="Random Forest"):
    """Genera figura comparativa CV vs Test externo para las 3 métricas."""
    bacterias = list(resultados_cv.keys())
    bacterias_cortas = [b.replace("_", " ").replace("Staphylococcus", "S.")
                        .replace("Escherichia", "E.").replace("Pseudomonas", "P.")
                        .replace("Klebsiella", "K.").replace("Acinetobacter", "A.")
                        for b in bacterias]

    metricas = ["roc_auc", "f1", "mcc"]
    titulos  = ["ROC-AUC", "F1-Score", "MCC"]
    x = np.arange(len(bacterias))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for i, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[i]
        cv_vals   = [resultados_cv[b][metrica] for b in bacterias]
        test_vals = [resultados_test[b][metrica] for b in bacterias]

        ax.bar(x - width/2, cv_vals,   width, label="Validación Cruzada",
               color="steelblue", alpha=0.8)
        ax.bar(x + width/2, test_vals, width, label="Test Externo",
               color="coral", alpha=0.8)

        ax.set_ylabel(titulo)
        ax.set_title(f"{titulo} — CV vs Test Externo")
        ax.set_xticks(x)
        ax.set_xticklabels(bacterias_cortas, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(0.3, 1.0)
        ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.4)

    plt.suptitle(f"{nombre_modelo} — Validación Cruzada vs Test Externo (Scaffold Split)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "cv_vs_test_externo.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {path}")


def plot_heatmap(resultados_cv_rf, resultados_test_rf,
                 resultados_cv_xgb, resultados_test_xgb):
    """Genera heatmap comparativo de métricas RF vs XGBoost."""
    bacterias = list(resultados_cv_rf.keys())
    bacterias_cortas = [b.replace("_", " ").replace("Staphylococcus", "S.")
                        .replace("Escherichia", "E.").replace("Pseudomonas", "P.")
                        .replace("Klebsiella", "K.").replace("Acinetobacter", "A.")
                        for b in bacterias]

    cols = ["CV AUC", "Test AUC", "CV F1", "Test F1", "CV MCC", "Test MCC"]

    data_rf = np.array([
        [resultados_cv_rf[b]["roc_auc"],  resultados_test_rf[b]["roc_auc"],
         resultados_cv_rf[b]["f1"],       resultados_test_rf[b]["f1"],
         resultados_cv_rf[b]["mcc"],      resultados_test_rf[b]["mcc"]]
        for b in bacterias
    ])

    data_xgb = np.array([
        [resultados_cv_xgb[b]["roc_auc"],  resultados_test_xgb[b]["roc_auc"],
         resultados_cv_xgb[b]["f1"],       resultados_test_xgb[b]["f1"],
         resultados_cv_xgb[b]["mcc"],      resultados_test_xgb[b]["mcc"]]
        for b in bacterias
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, cmap in zip(
        axes,
        [data_rf, data_xgb],
        ["Random Forest", "XGBoost"],
        ["Blues", "Oranges"],
    ):
        sns.heatmap(data, annot=True, fmt=".3f", cmap=cmap,
                    xticklabels=cols, yticklabels=bacterias_cortas,
                    ax=ax, vmin=0.4, vmax=1.0,
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size": 11})
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Heatmap de Métricas — CV y Test Externo",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "heatmap_metricas.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {path}")


# 11. PIPELINE PRINCIPAL


def main():
    print("=" * 60)
    print("QSAR PIPELINE - Predicción de Actividad Antibacteriana")
    print("=" * 60)

    # ── PASO 1: Descargar datos ──────────────────────────────
    print("\n[1/6] Descargando datos de ChEMBL...")
    datasets_raw = {}
    for nombre, chembl_id in BACTERIAS.items():
        df = descargar_datos_chembl(chembl_id, nombre)
        datasets_raw[nombre] = df

    # ── PASO 2: Limpiar datasets ─────────────────────────────
    print("\n[2/6] Preprocesando datasets...")
    datasets_limpios = {}
    for nombre, df in datasets_raw.items():
        df_limpio = limpiar_dataset(df, nombre)
        df_limpio.to_csv(
            os.path.join(DATA_PROCESSED, f"{nombre}_clean.csv"), index=False
        )
        datasets_limpios[nombre] = df_limpio

    # ── PASO 3: Scaffold split ───────────────────────────────
    print("\n[3/6] Aplicando scaffold split...")
    splits = {}
    for nombre, df in datasets_limpios.items():
        train, test = scaffold_split(df)
        train.to_csv(os.path.join(DATA_PROCESSED, f"{nombre}_train.csv"), index=False)
        test.to_csv(os.path.join(DATA_PROCESSED, f"{nombre}_test.csv"), index=False)
        splits[nombre] = (train, test)
        print(f"  {nombre}: train={len(train)}, test={len(test)}")

    # ── PASO 4: Calcular fingerprints ────────────────────────
    print("\n[4/6] Calculando Morgan fingerprints (ECFP4)...")
    fingerprints = {}
    for nombre, (train, test) in splits.items():
        X_train, y_train = calcular_fingerprints(train)
        X_test,  y_test  = calcular_fingerprints(test)
        np.save(os.path.join(DATA_FP, f"{nombre}_X_train.npy"), X_train)
        np.save(os.path.join(DATA_FP, f"{nombre}_y_train.npy"), y_train)
        np.save(os.path.join(DATA_FP, f"{nombre}_X_test.npy"),  X_test)
        np.save(os.path.join(DATA_FP, f"{nombre}_y_test.npy"),  y_test)
        fingerprints[nombre] = (X_train, y_train, X_test, y_test)
        print(f"  {nombre}: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # ── PASO 5: Entrenar y evaluar modelos ───────────────────
    print("\n[5/6] Entrenando y evaluando modelos...")

    resultados_cv_rf   = {}
    resultados_cv_xgb  = {}
    resultados_test_rf  = {}
    resultados_test_xgb = {}

    for nombre, (X_train, y_train, X_test, y_test) in fingerprints.items():
        print(f"\n  {nombre}:")

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )
        cv_rf = evaluar_con_cv(rf, X_train, y_train)
        test_rf = evaluar_en_test(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            X_train, y_train, X_test, y_test
        )
        resultados_cv_rf[nombre]   = cv_rf
        resultados_test_rf[nombre] = test_rf
        print(f"    RF  CV:   AUC={cv_rf['roc_auc']:.3f}±{cv_rf['roc_auc_std']:.3f}  "
              f"F1={cv_rf['f1']:.3f}  MCC={cv_rf['mcc']:.3f}")
        print(f"    RF  Test: AUC={test_rf['roc_auc']:.3f}  "
              f"F1={test_rf['f1']:.3f}  MCC={test_rf['mcc']:.3f}")

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="logloss", verbosity=0
        )
        cv_xgb = evaluar_con_cv(xgb, X_train, y_train)
        test_xgb = evaluar_en_test(
            XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                          random_state=RANDOM_STATE, n_jobs=-1,
                          eval_metric="logloss", verbosity=0),
            X_train, y_train, X_test, y_test
        )
        resultados_cv_xgb[nombre]   = cv_xgb
        resultados_test_xgb[nombre] = test_xgb
        print(f"    XGB CV:   AUC={cv_xgb['roc_auc']:.3f}±{cv_xgb['roc_auc_std']:.3f}  "
              f"F1={cv_xgb['f1']:.3f}  MCC={cv_xgb['mcc']:.3f}")
        print(f"    XGB Test: AUC={test_xgb['roc_auc']:.3f}  "
              f"F1={test_xgb['f1']:.3f}  MCC={test_xgb['mcc']:.3f}")

    # Guardar resultados
    pd.DataFrame(resultados_cv_rf).T.to_csv(
        os.path.join(DATA_PROCESSED, "resultados_cv_rf.csv")
    )
    pd.DataFrame(resultados_cv_xgb).T.to_csv(
        os.path.join(DATA_PROCESSED, "resultados_cv_xgb.csv")
    )
    pd.DataFrame(resultados_test_rf).T.to_csv(
        os.path.join(DATA_PROCESSED, "resultados_test_rf.csv")
    )
    pd.DataFrame(resultados_test_xgb).T.to_csv(
        os.path.join(DATA_PROCESSED, "resultados_test_xgb.csv")
    )

    # ── PASO 5b: Tuning con Optuna (A. baumannii) ────────────
    print("\n  Tuning XGBoost con Optuna para Acinetobacter_baumannii...")
    nombre_tuning = "Acinetobacter_baumannii"
    X_tr, y_tr, _, _ = fingerprints[nombre_tuning]
    mejores_params, mejor_auc = tunear_xgboost(X_tr, y_tr, n_trials=N_TRIALS)

    print(f"  Mejor AUC: {mejor_auc:.3f}")
    print(f"  Mejores params: {mejores_params}")

    with open(os.path.join(DATA_PROCESSED, "mejores_params_xgb.json"), "w") as f:
        json.dump({"params": mejores_params, "roc_auc": mejor_auc}, f, indent=2)

    #  PASO 6: Visualizaciones 
    print("\n[6/6] Generando figuras...")
    plot_cv_vs_test(resultados_cv_rf, resultados_test_rf)
    plot_heatmap(resultados_cv_rf, resultados_test_rf,
                 resultados_cv_xgb, resultados_test_xgb)

    #  RESUMEN FINAL 
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"\n{'Bacteria':<25} {'RF CV':>8} {'RF Test':>8} {'Δ AUC':>7} "
          f"{'XGB CV':>8} {'XGB Test':>9}")
    print("-" * 65)

    for nombre in BACTERIAS:
        print(
            f"{nombre.replace('_', ' '):<25} "
            f"{resultados_cv_rf[nombre]['roc_auc']:>8.3f} "
            f"{resultados_test_rf[nombre]['roc_auc']:>8.3f} "
            f"{resultados_test_rf[nombre]['roc_auc'] - resultados_cv_rf[nombre]['roc_auc']:>7.3f} "
            f"{resultados_cv_xgb[nombre]['roc_auc']:>8.3f} "
            f"{resultados_test_xgb[nombre]['roc_auc']:>9.3f}"
        )

    print("\nPipeline completado. Resultados en data/processed/ y figures/")


if __name__ == "__main__":
    main()
