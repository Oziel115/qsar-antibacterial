# Predicción de Actividad Antibacteriana Mediante Machine Learning y QSAR

**Oziel A. Carrión · Abimael Guzmán**  
Universidad Autónoma de Chihuahua · Facultad de Medicina y Ciencias Biomédicas

---

## Descripción

Pipeline completo de QSAR (*Quantitative Structure-Activity Relationship*) para predecir actividad antibacteriana en cinco patógenos de prioridad crítica según la lista WHO 2024, usando datos públicos de ChEMBL.

Se comparan Random Forest y XGBoost con Morgan fingerprints (ECFP4), evaluados con validación cruzada estratificada de 5 folds y con un test externo riguroso basado en scaffold split de Murcko.

---

## Patógenos estudiados

| Bacteria | ChEMBL ID | Compuestos | Activos |
|----------|-----------|-----------|---------|
| *Staphylococcus aureus* | CHEMBL352 | 48,080 | 53.0% |
| *Escherichia coli* | CHEMBL354 | 31,448 | 40.1% |
| *Pseudomonas aeruginosa* | CHEMBL348 | 19,945 | 34.9% |
| *Klebsiella pneumoniae* | CHEMBL350 | 11,662 | 45.4% |
| *Acinetobacter baumannii* | CHEMBL614425 | 4,534 | 43.8% |

**Total: 115,669 compuestos únicos**

---

## Resultados principales

| Bacteria | RF CV AUC | RF Test AUC | XGB CV AUC | XGB Test AUC |
|----------|-----------|-------------|-----------|--------------|
| *S. aureus* | 0.908 | 0.826 | 0.856 | 0.783 |
| *E. coli* | 0.920 | 0.819 | 0.876 | 0.770 |
| *P. aeruginosa* | 0.917 | 0.806 | 0.882 | 0.768 |
| *K. pneumoniae* | 0.927 | 0.832 | 0.909 | 0.802 |
| *A. baumannii* | 0.876 | 0.807 | 0.864 | 0.793 |
| **Promedio** | **0.910** | **0.818** | **0.877** | **0.783** |

**Hallazgo clave:** La validación cruzada sobreestima el rendimiento real en promedio **0.091 ROC-AUC** al compararse con scaffold split externo, evidenciando la importancia de una evaluación rigurosa en QSAR.

---

## Metodología

```
ChEMBL API
    ↓
Descarga de MIC (Minimum Inhibitory Concentration)
    ↓
Limpieza: conversión de unidades (nM → µg/mL), mediana para duplicados
    ↓
Clasificación: MIC ≤ 8 µg/mL = activo (Ebejer et al. 2016)
    ↓
Morgan Fingerprints: radio=2, nBits=2048 (ECFP4, Rogers & Hahn 2010)
    ↓
Scaffold Split (Murcko): 80% train / 20% test
    ↓
Validación Cruzada: StratifiedKFold 5 folds sobre train set
    ├── Random Forest (n_estimators=100)
    └── XGBoost + tuning con Optuna (50 trials, optimización bayesiana)
    ↓
Evaluación final en test externo (una sola vez)
    ↓
Métricas: ROC-AUC, F1-Score, MCC
```

---

## Estructura del repositorio

```
qsar-antibacterial/
├── README.md
├── requirements.txt
├── qsar_pipeline.py          ← Script principal completo
├── notebooks/
│   ├── 01_descarga_datos.ipynb
│   └── 02_modelos.ipynb
├── data/
│   └── processed/
│       ├── resultados_cv_rf.csv
│       ├── resultados_cv_xgb.csv
│       ├── resultados_test_rf.csv
│       └── resultados_test_xgb.csv
└── figures/
    ├── cv_vs_test_externo.png
    └── heatmap_metricas.png
```

> **Nota:** Los datos crudos de ChEMBL no se incluyen en el repositorio por su tamaño. El script los descarga automáticamente al ejecutarse.

---

## Instalación y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/qsar-antibacterial.git
cd qsar-antibacterial
```

### 2. Crear entorno virtual

```bash
conda create -n qsar python=3.10
conda activate qsar
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el pipeline completo

```bash
python qsar_pipeline.py
```

O abrir los notebooks en orden:

```bash
jupyter notebook notebooks/01_descarga_datos.ipynb
jupyter notebook notebooks/02_modelos.ipynb
```

---

## Dependencias principales

```
rdkit>=2023.9
pandas>=2.0
numpy>=1.21
scikit-learn>=1.3
xgboost>=2.0
optuna>=3.0
chembl-webresource-client>=0.10
matplotlib>=3.7
seaborn>=0.12
deepchem>=2.7
```

---

## Figuras

### Validación Cruzada vs Test Externo (Scaffold Split)

> *[Insertar figura cv_vs_test_externo.png]*

La caída consistente entre validación cruzada y test externo confirma que la validación cruzada sobreestima el rendimiento cuando los datos se dividen aleatoriamente.

### Heatmap de Métricas

> *[Insertar figura heatmap_metricas.png]*

Random Forest supera consistentemente a XGBoost en todas las bacterias y todas las métricas.

---

## Referencias

1. Ebejer JP, Charlton MH, Finn PW. Are the physicochemical properties of antibacterial compounds really different from other drugs? *J Cheminform*. 2016;8:30. DOI: [10.1186/s13321-016-0143-5](https://doi.org/10.1186/s13321-016-0143-5)

2. Rogers D, Hahn M. Extended-Connectivity Fingerprints. *J Chem Inf Model*. 2010;50(5):742–754. DOI: [10.1021/ci100050t](https://doi.org/10.1021/ci100050t)

3. Li WX, et al. Screening of antibacterial compounds with novel structure from the FDA approved drugs using machine learning methods. *Aging*. 2022;14(3):1448–1465. DOI: [10.18632/aging.203887](https://doi.org/10.18632/aging.203887)

4. Gaulton A, et al. The ChEMBL database in 2017. *Nucleic Acids Res*. 2017;45(D1):D945–D954. DOI: [10.1093/nar/gkw1074](https://doi.org/10.1093/nar/gkw1074)

5. Akiba T, et al. Optuna: A Next-generation Hyperparameter Optimization Framework. *Proc. 25th ACM SIGKDD*. 2019. DOI: [10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701)

6. Landrum G. RDKit: Open-Source Cheminformatics Software. 2006. [rdkit.org](https://www.rdkit.org)


