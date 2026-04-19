# CRISPR Off-Target Prediction Project - Documentation

## Executive Summary

Questo progetto sviluppa modelli di machine learning per predire l'attività off-target del sistema CRISPR-Cas9. Il sistema confronta diverse codifiche di sequenze DNA (8 encoding methods) e diversi backend di modelli (XGBoost, CatBoost, Decision Tree) utilizzando validazione incrociata a 10 fold e test statistici rigorosi.

---

## 1. Problem Statement

### 1.1 Il Task di Machine Learning

Il progetto affronta la predizione degli **off-target del CRISPR-Cas9**, formulando il problema in due modi:

| Formulazione | Descrizione | Target Variable |
|--------------|-------------|-----------------|
| **Classificazione** | Distinguere siti off-target attivi (1) da inattivi (0) | Label binaria |
| **Regressione** | Predire il numero di reads (intensità di taglio) | Read counts |

### 1.2 Cos'è la Predizione Off-Target?

CRISPR-Cas9 è uno strumento di editing genomico che può tagliare il DNA in posizioni specifiche. Tuttavia, può anche tagliare in siti **off-target** - posizioni genomiche simili ma non identiche al target desiderato.

**Perché è importante:**
- **Sicurezza terapeutica**: Gli off-target possono causare mutazioni indesiderate
- **Applicazioni cliniche**: Necessario identificare potenziali effetti collaterali prima del trattamento
- **Design di sgRNA**: Ottimizzare le guide RNA per minimizzare gli off-target

### 1.3 Sfide del Problema

1. **Forte sbilanciamento delle classi**: Pochi siti positivi (attivi) vs moltissimi negativi
2. **Variabilità tra sgRNA**: Ogni guida RNA ha pattern off-target diversi
3. **Feature engineering**: Codificare efficacemente le sequenze DNA per il ML
4. **Generalizzazione**: Il modello deve funzionare su nuove sgRNA non viste in training

---

## 2. Datasets

### 2.1 Dataset Disponibili

| Dataset | Descrizione | Dimensione Approssimativa | Sorgente |
|---------|-------------|---------------------------|----------|
| **CHANGE-seq** | Off-target sperimentali validati | ~100K siti | Esperimenti CHANGE-seq |
| **GUIDE-seq** | Off-target sperimentali validati | ~50K siti | Esperimenti GUIDE-seq |

### 2.2 Struttura dei File

```
files/datasets/
├── CHANGE-seq.xlsx              # Dati raw CHANGE-seq
├── GUIDE-seq.xlsx               # Dati raw GUIDE-seq
├── include_on_targets/          # Dataset con on-target inclusi
│   ├── CHANGEseq_positive.csv   # Siti attivi (label=1)
│   ├── CHANGEseq_negative.csv   # Siti inattivi (label=0)
│   ├── GUIDEseq_positive.csv
│   └── GUIDEseq_negative.csv
└── exclude_on_targets/          # Dataset con on-target esclusi
    ├── CHANGEseq_positive.csv
    └── ...
```

### 2.3 Schema dei Dati

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| `chrom` | str | Cromosoma (chr1, chr2, ..., chrX, chrY) |
| `chromStart` | int | Posizione genomica di inizio |
| `strand` | str | Filamento DNA (+ o -) |
| `offtarget_sequence` | str | Sequenza off-target di 23bp |
| `target` | str | Sequenza sgRNA di 23bp |
| `distance` | int | Distanza di Hamming dal target (0=on-target) |
| `label` | int | 1=attivo, 0=inattivo, -1=indefinito |
| `{dataset}_reads` | float | Read counts sperimentali |

### 2.4 Definizione di Positive e Negative Samples

**Campioni Positivi (label=1):**
- Off-target con read count > soglia (tipicamente 100)
- Siti di taglio sperimentalmente rilevati
- Opzionale: esclusione degli on-target (distance=0)

**Campioni Negativi (label=0):**
- Potenziali off-target predetti da CAS-OFFinder ma NON rilevati sperimentalmente
- Filtrati per rimuovere:
  - Sequenze con carattere 'N'
  - Off-target su cromosomi non nell'esperimento
  - Duplicati per lo stesso target
  - On-target (distance=0)

**Campioni Indefiniti (label=-1):**
- Off-target con read count ≤ soglia (attività ambigua)
- Creati solo quando specificata una read_threshold

### 2.5 sgRNA (Target) nel Dataset

Ogni dataset contiene ~14-20 diverse guide RNA (sgRNA). La validazione incrociata divide le sgRNA in fold, garantendo che il modello sia testato su sgRNA non viste durante il training.

---

## 3. Data Preprocessing

### 3.1 Preprocessing delle Sequenze

**Mascheramento della regione PAM:**
```python
# Le ultime 3 posizioni contengono la PAM (es. NGG)
# Posizione 21-22 sono sostituite per uniformare
dataset_df['target'] = dataset_df['target'].apply(lambda s: s[:-3] + 'N' + s[-2:])
```

**Pulizia delle Sequenze:**
- Conversione a uppercase
- Filtraggio sequenze con 'N' o gap '-' (a meno che non siano gestiti dall'encoding)

### 3.2 Trasformazioni della Variabile Target (Regressione)

Per i modelli di regressione, i read counts sono trasformati prima del training:

| Trasformazione | Formula Forward | Formula Inverse | Uso |
|----------------|-----------------|-----------------|-----|
| `no_trans` | y = x | x = y | Nessuna trasformazione |
| `ln_x_plus_one_trans` | y = log(x + 1) | x = exp(y) - 1 | **Default**, riduce skewness |
| `ln_x_plus_one_and_max_trans` | y = MaxAbsScale(log(x+1)) | Composita | Normalizza anche la scala |
| `standard_trans` | y = (x - μ) / σ | x = σ*y + μ | Z-score normalization |
| `max_trans` | y = x / max(\|x\|) | x = y * max(\|x\|) | Scaling [-1, 1] |
| `box_cox_trans` | Power transform | Inverse Box-Cox | Normalizza distribuzione |
| `yeo_johnson_trans` | Yeo-Johnson | Inverse YJ | Gestisce anche valori negativi |

**Opzioni di Fitting:**
- `trans_all_fold=False` (default): Trasformazione applicata per sgRNA individualmente
- `trans_all_fold=True`: Trasformazione applicata all'intero fold
- `trans_only_positive=True`: Fit solo sui campioni positivi

### 3.3 Gestione dello Sbilanciamento

**Sample Weights (Pesi dei Campioni):**
```python
def build_sampleweight(y_values):
    """Peso inversamente proporzionale alla frequenza della classe"""
    vec = np.zeros(len(y_values))
    for values_class in np.unique(y_values):
        vec[y_values == values_class] = np.sum(y_values != values_class) / len(y_values)
    return vec
```

**Dataset Bilanciato (opzionale):**
- Se `balanced=True`: Per ogni target, i negativi sono sottocampionati per eguagliare i positivi
- Default: `balanced=False` (usa tutti i negativi - naturalmente sbilanciato)

---

## 4. Feature Engineering / Encodings

### 4.1 Nucleotide Position Mapping

Tutti gli encoding utilizzano il **confronto pairwise** tra la sequenza target e quella off-target:

**Mapping Standard (4 nucleotidi):**
```
Nucleotidi: {A, C, G, T}
Combinazioni: 4×4 = 16 possibili coppie
Mapping: {('A','A'): (0,0), ('A','C'): (0,1), ..., ('T','T'): (3,3)}
```

**Mapping con Bulges (5 caratteri):**
```
Caratteri: {A, C, G, T, -}
Combinazioni: 5×5 = 25 possibili coppie
Include inserzioni/delezioni
```

### 4.2 Metodi di Encoding

#### **1. NPM (Neural Position Matrix)**

| Proprietà | Valore |
|-----------|--------|
| **Output** | 23 posizioni × 4×4 matrice = **368 features** |
| **Con distance** | 369 features |

**Processo:**
- Per ogni posizione (1-23), crea una matrice binaria 4×4
- `Matrix[i,j] = 1` dove (target_nuc, offtarget_nuc) mappa a (i,j)
- Flatten di tutte le 23 matrici
- Rappresentazione spaziale dei mismatch

#### **2. OneHot (One-Hot Encoding Standard)**

| Proprietà | Valore |
|-----------|--------|
| **Output** | 23 posizioni × 4 canali = **92 features** |
| **Con distance** | 93 features |

**Processo:**
```
Target:    A → [1,0,0,0], C → [0,1,0,0], G → [0,0,1,0], T → [0,0,0,1], N → [0,0,0,0]
Offtarget: encoding analogo
Combinazione: OR logico tra target e offtarget
```

#### **3. OneHot5Channel (One-Hot con Direzione)**

| Proprietà | Valore |
|-----------|--------|
| **Output** | 23 posizioni × 5 canali = **115 features** |
| **Con distance** | 116 features |

**Processo:**
- OneHot standard + 1 bit di direzione per posizione
- Direction bit = 0 se target ≤ offtarget (lessicograficamente)
- Direction bit = 1 se target > offtarget

#### **4. OneHotVstack (One-Hot Impilato)**

| Proprietà | Valore |
|-----------|--------|
| **Output** | 23 posizioni × 8 canali = **184 features** |
| **Con distance** | 185 features |

**Processo:**
- Concatenazione verticale (non OR) degli encoding
- Target one-hot: 4 canali
- Offtarget one-hot: 4 canali
- Mantiene le sequenze separate

#### **5. kmer (K-mer Binary Encoding)**

| Proprietà | Valore |
|-----------|--------|
| **Output** | (23 - k + 1) × k features (k=3: **63 features**) |
| **Con distance** | 64 features |

**Processo:**
```
k=3 (3-mers sovrapposti):
- Estrae k-mer sovrapposti da entrambe le sequenze
- Confronto position-wise: '1' per match, '0' per mismatch
Esempio: "ACG" vs "ACT" → "110"
```

#### **6. LabelEncodingPairwise (Encoding Ordinale)**

| Proprietà | Valore |
|-----------|--------|
| **Output** | 23 posizioni = **23 features** |
| **Con distance** | 24 features |

**Processo:**
```
16 valori ordinali:
('A','A')→0, ('A','C')→1, ('A','G')→2, ..., ('T','T')→15
'N' è trattato come l'altro nucleotide
```

#### **7. bulges (Encoding con Gap)**

| Proprietà | Valore |
|-----------|--------|
| **Output** | 23 posizioni × 5×5 matrice = **575 features** |
| **Con distance** | 576 features |

**Processo:**
- Simile a NPM ma include gap/bulges
- Nucleotidi: {A, C, G, T, -}
- Matrice 5×5 per posizione (25 combinazioni)
- Cattura inserzioni/delezioni

#### **8. MM (Match-Mismatch Binary)**

| Proprietà | Valore |
|-----------|--------|
| **Output** | 23 posizioni = **23 features** |
| **Con distance** | 24 features |

**Processo:**
```
Per ogni posizione:
- 0 se nucleotidi corrispondono
- 1 se nucleotidi differiscono
- -1 se uno è 'N' (sconosciuto)
```

### 4.3 Distance Feature

- **Descrizione**: Distanza di Hamming tra target e off-target
- **Inclusione**: Opzionale (`include_distance_feature=True/False`)
- **Posizione**: Aggiunta come ultima colonna della feature matrix

### 4.4 Riepilogo Dimensionalità

| Encoding | Features Base | Con Distance |
|----------|---------------|--------------|
| NPM | 368 | 369 |
| OneHot | 92 | 93 |
| OneHot5Channel | 115 | 116 |
| OneHotVstack | 184 | 185 |
| kmer (k=3) | 63 | 64 |
| LabelEncodingPairwise | 23 | 24 |
| bulges | 575 | 576 |
| MM | 23 | 24 |

---

## 5. Modelli di Machine Learning

### 5.1 XGBoost (Extreme Gradient Boosting)

**Classi:**
- Classificatore: `xgb.XGBClassifier`
- Regressore: `xgb.XGBRegressor`

**Iperparametri per Encoding:**

| Encoding | max_depth | learning_rate | n_estimators | subsample | colsample_bytree |
|----------|-----------|---------------|--------------|-----------|------------------|
| NPM | 12 | 0.1 | 500 | 1.0 | 1.0 |
| OneHot | 12 | 0.2 | 1000 | 1.0 | 0.8 |
| OneHot5Channel | 12 | 0.1 | 1000 | 1.0 | 1.0 |
| kmer | 12 | 0.2 | 1000 | 1.0 | 0.8 |
| LabelEncodingPairwise | 12 | 0.2 | 1000 | 1.0 | 0.8 |
| Altri | 12 | 0.2 | 1000 | 1.0 | 0.8 |

**Parametri Comuni:**
```python
{
    'nthread': 55,              # Multi-threading
    'random_state': 42,         # Riproducibilità
    'eval_metric': 'logloss',   # Per classificazione
}
```

### 5.2 CatBoost (Categorical Boosting)

**Classi:**
- Classificatore: `CatBoostClassifier`
- Regressore: `CatBoostRegressor`

**Iperparametri (Tutti gli Encoding):**
```python
{
    'depth': 12,                      # Profondità albero
    'learning_rate': 0.05,            # Learning rate
    'iterations': 3000,               # Round di boosting
    'l2_leaf_reg': 9,                 # Regolarizzazione L2
    'subsample': 0.8,                 # Row subsampling
    'bootstrap_type': 'Bernoulli',    # Tipo bootstrap
    'thread_count': -1,               # Auto (tutti i thread)
    'verbose': 100,                   # Verbosità
    'random_seed': 42,                # Riproducibilità
    'task_type': 'GPU',               # O 'CPU'
}
```

**Early Stopping:**
```python
{
    'early_stopping_rounds': 120,
    'use_best_model': True,
}
```
- Split 80/20 train/validation per early stopping

### 5.3 Decision Tree (Baseline)

**Classi:**
- Classificatore: `DecisionTreeClassifier`
- Regressore: `DecisionTreeRegressor`

**Iperparametri Ottimizzati:**
```python
{
    'splitter': 'random',          # Split casuali
    'min_samples_split': 55,       # Min campioni per split
    'min_samples_leaf': 17,        # Min campioni per foglia
    'max_features': 'log2',        # Features per split
    'max_depth': 20,               # Profondità massima
    'ccp_alpha': 0.011,            # Pruning
    'random_state': 42,            # Riproducibilità
    'criterion': 'log_loss',       # Per classificazione
    # 'criterion': 'absolute_error', # Per regressione
}
```

### 5.4 Confronto Modelli

| Aspetto | XGBoost | CatBoost | Decision Tree |
|---------|---------|----------|---------------|
| **Tipo** | Gradient Boosting | Gradient Boosting | Singolo Albero |
| **Complessità** | Alta | Alta | Bassa |
| **Velocità Training** | Veloce | Più lento | Molto veloce |
| **GPU Support** | Sì | Sì (nativo) | No |
| **Feature Categoriche** | Manuale | Nativo | Manuale |
| **Early Stopping** | Sì | Sì | No |
| **Interpretabilità** | Media | Media | Alta |

---

## 6. Training Pipeline

### 6.1 K-Fold Cross-Validation

**Struttura della Validazione:**
```
Input: Tutte le sgRNA, Dataset Positive/Negative
K = 10 fold

Per ogni fold i (0..9):
  ├─ Test targets: sgRNA del fold i
  ├─ Train targets: sgRNA dei restanti 9 fold
  ├─ Split train: 80% training, 20% validation (per early stopping)
  ├─ Train model_i
  ├─ Evaluate model_i su test targets
  └─ Save model_i e risultati

Aggregazione: Metriche aggregate su tutti i fold
```

**Caratteristiche:**
- **Split per sgRNA**: Ogni sgRNA è assegnata a un solo fold
- **Nessun data leakage**: Test sgRNA mai viste in training
- **Stratificazione**: Distribuzione bilanciata delle classi nel validation set

### 6.2 Procedura di Training Dettagliata

**Step 1: Caricamento Dati**
```python
targets = load_order_sg_rnas(data_type)  # Lista ordinata delle sgRNA
positive_df = pd.read_csv(f'{data_type}_positive.csv')
negative_df = pd.read_csv(f'{data_type}_negative.csv')
```

**Step 2: Creazione Fold Sets**
```python
for fold_index in range(k_fold_number):
    # Divide targets in train/test
    test_targets = targets[fold_start:fold_end]
    train_targets = [t for t in targets if t not in test_targets]
    
    # Filtra positive/negative per fold
    positive_df_train = positive_df[positive_df['target'].isin(train_targets)]
    negative_df_train = negative_df[negative_df['target'].isin(train_targets)]
```

**Step 3: Feature Building**
```python
# Build features per il training set
positive_features = build_sequence_features(
    positive_df_train, 
    nucleotides_to_position_mapping,
    encoding='OneHot',
    include_distance_feature=True,
    include_sequence_features=True
)
negative_features = build_sequence_features(negative_df_train, ...)

# Concatena
X_train_full = np.concatenate([negative_features, positive_features])
y_train_full = np.concatenate([np.zeros(len(negative_features)), 
                                np.ones(len(positive_features))])
```

**Step 4: Train/Validation Split**
```python
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, 
    random_state=42, 
    stratify=y_train_full  # Per classificazione
)
```

**Step 5: Training del Modello**
```python
if model_backend == 'xgboost':
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=build_sampleweight(y_train),
        eval_set=[(X_val, y_val)],
        verbose=100
    )

elif model_backend == 'catboost':
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=build_sampleweight(y_train),
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=120
    )
```

**Step 6: Salvataggio Modello**
```python
# Path pattern
model_path = f"files/models_10fold/new_models/{data_type}/include_on_targets/"
model_path += f"{model_type}/{encoding}/"
model_path += f"{data_type}_{model_type}_{backend}_model_fold_{fold_index}"
model_path += f"_with_distance_imbalanced_with_{encoding}Encoding.json"

# Save
if model_backend == 'xgboost':
    model.save_model(model_path)
elif model_backend == 'catboost':
    model.save_model(model_path)
elif model_backend == 'decision_tree':
    joblib.dump(model, model_path.replace('.json', '.joblib'))
```

### 6.3 Struttura File dei Modelli

```
files/models_10fold/new_models/
├── CHANGEseq/
│   └── include_on_targets/
│       ├── classifier/
│       │   ├── NPM/
│       │   │   ├── CHANGEseq_classifier_xgb_model_fold_0_with_distance_imbalanced.json
│       │   │   ├── CHANGEseq_classifier_xgb_model_fold_1_with_distance_imbalanced.json
│       │   │   └── ...
│       │   ├── OneHot/
│       │   └── ...
│       └── regression_with_negatives/
│           └── ...
└── GUIDEseq/
    └── ...
```

---

## 7. Evaluation

### 7.1 Metriche di Classificazione

| Metrica | Formula | Range | Ottimo | Descrizione |
|---------|---------|-------|--------|-------------|
| **AUPR** | Area sotto curva Precision-Recall | [0, 1] | 1.0 | **Primaria** - Robusta a sbilanciamento |
| **AUC** | Area sotto curva ROC | [0, 1] | 1.0 | Capacità discriminativa generale |
| **Accuracy** | (TP+TN) / Total | [0, 1] | 1.0 | Correttezza generale |
| **Precision** | TP / (TP+FP) | [0, 1] | 1.0 | Quanto sono corretti i positivi predetti |
| **Recall** | TP / (TP+FN) | [0, 1] | 1.0 | Quanti positivi sono catturati |
| **F1-Score** | 2×(P×R)/(P+R) | [0, 1] | 1.0 | Media armonica P/R |
| **Pearson** | Correlazione lineare | [-1, 1] | 1.0 | Correlazione probabilità-reads |
| **Spearman** | Correlazione di rango | [-1, 1] | 1.0 | Correlazione monotonica |

### 7.2 Metriche di Regressione

| Metrica | Formula | Range | Ottimo | Descrizione |
|---------|---------|-------|--------|-------------|
| **Pearson** | Corr(y_trans, ŷ) | [-1, 1] | 1.0 | Su scala trasformata |
| **Pearson_after_inv_trans** | Corr(y, ŷ_inv) | [-1, 1] | 1.0 | Su scala originale (reads) |
| **Pearson_only_positives** | Corr sui positivi | [-1, 1] | 1.0 | Performance su siti attivi |
| **Spearman** | Rank corr su trasformata | [-1, 1] | 1.0 | Monotonica trasformata |
| **Spearman_after_inv_trans** | Rank corr originale | [-1, 1] | 1.0 | Monotonica originale |
| **RMSE** | √(mean((y_trans - ŷ)²)) | [0, ∞) | 0.0 | Errore scala trasformata |
| **RMSE_after_inv_trans** | √(mean((y - ŷ_inv)²)) | [0, ∞) | 0.0 | Errore scala reads |
| **reg_to_class_AUPR** | AUPR usando predizioni reg | [0, 1] | 1.0 | La regressione discrimina? |

### 7.3 Procedura di Evaluation

**Per-Target Evaluation:**
```python
for target in test_targets:
    # Estrai predizioni e labels per questo target
    mask = predictions_df['target'] == target
    y_true = predictions_df.loc[mask, 'label']
    y_pred_proba = predictions_df.loc[mask, 'predictions']
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calcola metriche
    metrics = {
        'target': target,
        'positives': (y_true == 1).sum(),
        'negatives': (y_true == 0).sum(),
        'aupr': average_precision_score(y_true, y_pred_proba),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'pearson': pearsonr(y_true, y_pred_proba)[0],
        'spearman': spearmanr(y_true, y_pred_proba)[0],
    }
    
    results.append(metrics)
```

### 7.4 Struttura Output dei Risultati

```
files/models_10fold/new_models/CHANGEseq/include_on_targets/results/
├── NPM/
│   ├── CHANGEseq_classifier_results_xgb_model_all_10_folds_with_distance_imbalanced.csv
│   ├── CHANGEseq_classifier_results_catboost_model_all_10_folds_with_distance_imbalanced.csv
│   ├── CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv
│   └── CHANGEseq_regression_with_negatives_results_catboost_model_all_10_folds_with_distance_imbalanced.csv
├── OneHot/
│   ├── CHANGEseq_classifier_results_xgb_model_all_10_folds_with_distance_imbalanced_with_OneHotEncoding.csv
│   └── ...
└── (altri encoding...)
```

**Formato CSV Risultati:**
```csv
target,positives,negatives,accuracy,auc,aupr,precision,recall,f1_score,pearson,spearman,...
GCTGT...,4,23676,0.998,0.999,0.817,0.167,1.0,0.286,0.420,0.023,...
GGTGC...,73,33734,0.997,0.995,0.374,0.394,0.384,0.389,0.479,0.080,...
...
All Targets,4523,498521,0.978,0.965,0.547,0.423,0.512,0.463,0.456,0.122,...  # Aggregato
```

---

## 8. Test Statistici

### 8.1 Framework di Testing

**File Principale:** `statistical_tests.py`

**Scopo:** Confrontare statisticamente:
1. **Backend**: XGBoost vs CatBoost per ogni encoding
2. **Encoding**: Tutti gli 8 encoding tra loro per ogni backend

### 8.2 Test Utilizzati

#### **Wilcoxon Signed-Rank Test**

| Proprietà | Valore |
|-----------|--------|
| **Tipo** | Non-parametrico, paired |
| **Quando** | Confronto di 2 metodi sugli stessi target |
| **Ipotesi nulla** | Le due distribuzioni sono uguali |
| **Robusto a** | Outliers, non-normalità |

```python
from scipy.stats import wilcoxon

statistic, p_value = wilcoxon(
    method1_data, method2_data,
    alternative='two-sided',
    zero_method='wilcox'
)
```

#### **Friedman Test**

| Proprietà | Valore |
|-----------|--------|
| **Tipo** | Non-parametrico, omnibus |
| **Quando** | Confronto di >2 metodi simultaneamente |
| **Ipotesi nulla** | Tutte le distribuzioni sono uguali |
| **Equivalente a** | ANOVA non-parametrica per misure ripetute |

```python
from scipy.stats import friedmanchisquare

statistic, p_value = friedmanchisquare(*data_arrays)
```

#### **Post-Hoc Pairwise con Correzione Bonferroni**

| Proprietà | Valore |
|-----------|--------|
| **Quando** | Dopo Friedman test significativo |
| **Correzione** | α_corrected = 0.05 / n_comparisons |
| **Con 8 encoding** | 28 confronti pairwise, α = 0.0018 |

### 8.3 Interpretazione dei Risultati

| P-value | Simbolo | Interpretazione |
|---------|---------|-----------------|
| p < 0.001 | *** | Evidenza molto forte |
| p < 0.01 | ** | Evidenza forte |
| p < 0.05 | * | Evidenza moderata |
| p ≥ 0.05 | ns | Non significativo |

**Effect Size (Cohen's d):**
| Cohen's d | Interpretazione |
|-----------|-----------------|
| \|d\| < 0.2 | Effetto piccolo |
| 0.2 ≤ \|d\| < 0.5 | Effetto medio |
| \|d\| ≥ 0.5 | Effetto grande |

### 8.4 Uso del Framework

```python
from statistical_tests import (
    run_backend_comparison,
    run_encoding_comparison,
    compare_two_methods
)

# Confronto XGBoost vs CatBoost per tutti gli encoding
backend_results = run_backend_comparison(
    encodings=['NPM', 'OneHot', 'kmer', ...],
    metric='aupr',
    data_type='CHANGEseq',
    model_type='classifier',
    with_distance=True
)

# Confronto tutti gli encoding per un backend
encoding_results = run_encoding_comparison(
    encodings=['NPM', 'OneHot', ...],
    backend='catboost',
    metric='aupr',
    data_type='CHANGEseq',
    model_type='classifier'
)
```

### 8.5 Output dei Test Statistici

```
files/
├── statistical_summary_classification.csv
├── statistical_summary_regression.csv
├── backend_comparison_aupr_classifier_CHANGEseq.csv
├── encoding_comparison_catboost_aupr_classifier_CHANGEseq.csv
└── encoding_comparison_xgb_aupr_classifier_CHANGEseq.csv
```

---

## 9. Struttura del Progetto

```
OffTargetPrevention/
├── main_train.py                    # Entry point per il training
├── main_test.py                     # Entry point per la valutazione
├── statistical_tests.py             # Framework test statistici
├── run_statistical_tests_example.py # Esempio uso test statistici
├── prepare_data.py                  # Preprocessing dataset
├── plot_data.py                     # Visualizzazioni
├── explainability.py                # SHAP analysis
├── compare_encodings.py             # Confronto encoding
│
├── SysEvalOffTarget_src/            # Moduli core
│   ├── train_utilities.py           # Funzioni di training
│   ├── test_utilities.py            # Funzioni di test
│   ├── utilities.py                 # Utility generiche
│   ├── general_utilities.py         # Costanti globali
│   └── encoding.py                  # Implementazione encoding
│
├── files/                           # Directory dati
│   ├── datasets/                    # Dataset raw e processati
│   │   ├── CHANGE-seq.xlsx
│   │   ├── GUIDE-seq.xlsx
│   │   ├── include_on_targets/
│   │   └── exclude_on_targets/
│   ├── models_10fold/               # Modelli salvati
│   │   └── new_models/
│   │       ├── CHANGEseq/
│   │       └── GUIDEseq/
│   └── encoding_comparison/         # Risultati confronto
│
├── Best_Models/                     # Migliori modelli per deploy
├── Supplementary_Tables/            # Tabelle supplementari
└── cache/                           # Cache features
```

---

## 10. Risultati Principali (da completare)

### 10.1 Confronto Backend (XGBoost vs CatBoost)

*Da completare con risultati dei test statistici*

| Encoding | Metrica | XGBoost | CatBoost | P-value | Winner |
|----------|---------|---------|----------|---------|--------|
| OneHot | AUPR | 0.515 | 0.547 | 4.58×10⁻⁹ | CatBoost*** |
| NPM | AUPR | 0.611 | 0.628 | TBD | TBD |
| ... | ... | ... | ... | ... | ... |

### 10.2 Confronto Encoding

*Da completare con risultati del Friedman test*

### 10.3 Migliori Configurazioni

*Da completare dopo analisi completa*

---

## 11. Guida all'Esecuzione

### 11.1 Training

```bash
# Training con CatBoost su tutti gli encoding
python main_train.py

# Oppure configurazione specifica
# Modificare i parametri in main_train.py nella sezione if __name__ == "__main__"
```

### 11.2 Evaluation

```bash
# Valutazione modelli
python main_test.py
```

### 11.3 Test Statistici

```bash
# Test rapido
python debug_comparison.py

# Analisi completa
python run_statistical_tests_example.py

# Tutte le analisi
python statistical_tests.py
```

---

## 12. Dipendenze

```
numpy
pandas
scikit-learn
xgboost
catboost
scipy
matplotlib
seaborn
joblib
openpyxl  # Per lettura Excel
shap      # Per explainability
```

---

## Appendice A: Glossario

| Termine | Definizione |
|---------|-------------|
| **sgRNA** | Single guide RNA - la sequenza guida per CRISPR |
| **PAM** | Protospacer Adjacent Motif - sequenza riconosciuta da Cas9 |
| **Off-target** | Sito genomico tagliato non intenzionalmente |
| **On-target** | Sito genomico target intenzionale |
| **Read count** | Numero di sequenze rilevate in un esperimento |
| **AUPR** | Area Under Precision-Recall curve |
| **Hamming distance** | Numero di posizioni con nucleotidi diversi |

---

## Appendice B: Riferimenti

*Da completare con riferimenti bibliografici*

1. CHANGE-seq paper
2. GUIDE-seq paper
3. XGBoost paper
4. CatBoost paper
5. Riferimenti su off-target prediction

---

*Documento generato automaticamente il 2026-03-29*
*Versione: 1.0*
