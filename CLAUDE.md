# ML Model Monitoring Dashboard — Guide de développement pour Claude Code

## Contexte et objectif

Système de monitoring de modèles ML en production qui surveille la dérive des données et des prédictions au fil du temps. Le système détecte automatiquement quand un modèle commence à se dégrader et génère des alertes avant que l'impact business soit visible.

Publié sur GitHub comme projet portfolio, déployé sur Streamlit Community Cloud.

**Ce que le projet démontre aux recruteurs :**
- Tu penses production, pas juste notebook
- Tu comprends que les modèles se dégradent dans le temps (data drift, concept drift)
- Tu maîtrises l'observabilité des systèmes ML — compétence rare chez les candidats

> ⚠️ **Instruction critique** : pour chaque librairie (Evidently, scikit-learn, SQLite, Streamlit), consulter la documentation officielle à jour avant d'écrire du code. Ne pas supposer que les APIs sont stables.

---

## Stack technique

| Composant | Technologie | Raison du choix |
|---|---|---|
| Détection de drift | Evidently AI | Standard industrie pour le monitoring ML, rapports riches |
| Modèles de référence | scikit-learn | Universellement reconnu, facile à remplacer par n'importe quel modèle |
| Stockage des métriques | SQLite | Zéro infrastructure, suffisant pour un portfolio, portable |
| Scheduling | APScheduler | Léger, pas besoin d'Airflow pour ce cas d'usage |
| Interface | Streamlit | Déploiement simple, visualisations interactives |
| Visualisations | Plotly | Graphiques interactifs, intégration native Streamlit |
| Déploiement | Streamlit Community Cloud | Gratuit, connecté directement au repo GitHub |

---

## Structure du projet

```
ml-monitoring-dashboard/
├── app.py                          # Dashboard Streamlit principal
├── monitoring/
│   ├── __init__.py
│   ├── drift_detector.py           # Détection data drift et concept drift
│   ├── performance_tracker.py      # Suivi des métriques de performance
│   └── alerting.py                 # Logique d'alertes et seuils
├── data/
│   ├── __init__.py
│   ├── simulator.py                # Simulation de dérive des données (démo)
│   └── storage.py                  # Lecture/écriture SQLite
├── models/
│   ├── __init__.py
│   ├── baseline_model.py           # Wrapper modèle de référence
│   └── model_registry.py           # Métadonnées et versioning du modèle
├── core/
│   ├── __init__.py
│   └── config.py                   # Configuration et seuils d'alerte
├── tests/
│   ├── test_drift_detector.py
│   ├── test_performance_tracker.py
│   └── test_simulator.py
├── notebooks/
│   └── 01_baseline_training.ipynb  # Entraînement du modèle de référence
├── artifacts/
│   ├── baseline_model.joblib       # Modèle sauvegardé
│   ├── reference_data.csv          # Dataset d'entraînement (reference Evidently)
│   └── model_metadata.json         # Métriques et métadonnées de référence
├── monitoring.db                   # Base SQLite pré-remplie (committée)
├── run_monitoring.py               # Script de monitoring quotidien
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Concepts clés à maîtriser avant de coder

Lire et comprendre ces trois concepts — ils sont le cœur du projet.

**Data Drift** : la distribution des features en entrée change par rapport à ce que le modèle a vu à l'entraînement. Exemple : les clients d'une banque commencent à avoir des revenus moyens plus élevés. Le modèle n'a pas été entraîné sur ce segment — ses prédictions se dégradent silencieusement.

**Concept Drift** : la relation entre les features et la target change. Exemple : les comportements qui prédisaient le churn en 2022 ne prédisent plus le churn en 2024. Le modèle reçoit les mêmes inputs mais devrait donner des outputs différents.

**Reference Dataset vs Current Dataset** : Evidently fonctionne en comparant un dataset de référence (les données d'entraînement) à un dataset courant (les données de production récentes). Comprendre ce pattern est indispensable avant d'écrire une ligne de code Evidently.

---

## ✅ Étape 1 — Setup et données de démonstration (TERMINÉE)

Le projet n'a pas de "vrai" dataset de production — il faut simuler la dérive pour que le dashboard soit démontrable. C'est la première chose à construire.

**`core/config.py`** : constantes globales du projet via `pydantic-settings`. Définir ici les seuils d'alerte (drift_score > 0.15 = warning, > 0.30 = critical), les fenêtres temporelles de monitoring (7 jours, 30 jours), les chemins vers les artifacts et la base SQLite, et le nombre minimum de samples pour calculer des métriques fiables (30).

**`data/simulator.py`** : générateur de données synthétiques qui simule trois scénarios distincts.

Scénario 1 — **Stable** : données qui restent dans la même distribution que le dataset d'entraînement. Sert de baseline pour vérifier que les alertes ne se déclenchent pas à tort.

Scénario 2 — **Gradual Drift** : dérive progressive sur 30 jours. La moyenne de certaines features glisse lentement. Simule ce qui arrive en production réelle — rarement brutal, souvent insidieux.

Scénario 3 — **Sudden Drift** : changement abrupt à une date précise. Simule un événement externe (crise économique, changement réglementaire, bug de données upstream).

Exposer une fonction `generate_production_batch(scenario, date, n_samples) -> pd.DataFrame` qui retourne un batch de données pour une date donnée avec le niveau de dérive correspondant au scénario. Les features doivent avoir des noms business réalistes (feature_age, feature_income, feature_tenure, etc.) plutôt que feature_0, feature_1.

**`data/storage.py`** : interface SQLite avec trois tables.

```
predictions_log :
  id, timestamp, feature_values (JSON), prediction, probability, actual_label (nullable)

drift_reports :
  id, timestamp, window_start, window_end, drift_score, drifted_features (JSON),
  report_type (data_drift | concept_drift), severity (ok | warning | critical)

performance_metrics :
  id, timestamp, window_start, window_end, accuracy, precision, recall, f1, roc_auc,
  n_samples
```

Exposer des fonctions typées : `log_prediction(...)`, `save_drift_report(...)`, `save_performance_metrics(...)`, `get_predictions_window(start, end) -> pd.DataFrame`, `get_drift_history(days) -> list[dict]`.

**✅ Validation** : le simulateur génère des DataFrames cohérents pour les trois scénarios. Les données s'écrivent et se lisent correctement depuis SQLite. Les trois scénarios produisent des distributions visuellement différentes avec un simple `df.describe()`.

---

## ✅ Étape 2 — Modèle de référence (TERMINÉE)

**`notebooks/01_baseline_training.ipynb`** : entraîner un modèle de classification binaire sur le dataset synthétique "stable". Utiliser `make_classification` de scikit-learn avec des features nommées de façon business. Sauvegarder dans `artifacts/` : le modèle avec `joblib`, le dataset d'entraînement en CSV (reference dataset Evidently), et les métriques de performance en JSON.

**`models/baseline_model.py`** : wrapper autour du modèle sauvegardé. Expose `predict(X) -> np.ndarray` et `predict_proba(X) -> np.ndarray`. Charge le modèle depuis `artifacts/baseline_model.joblib` au démarrage. Gère le cas où le fichier n'existe pas avec une erreur explicite et un message indiquant de lancer le notebook d'abord.

**`models/model_registry.py`** : lit `artifacts/model_metadata.json` et expose `get_current_model_info() -> dict` avec version, date d'entraînement, métriques de référence, et chemin vers le fichier modèle.

**✅ Validation** : le notebook s'exécute de bout en bout sans erreur. Le modèle charge correctement. `predict_proba` retourne des probabilités entre 0 et 1. Les trois fichiers artifacts sont générés dans `artifacts/`.

---

## ✅ Étape 3 — Détection de drift (TERMINÉE)

Lire la documentation Evidently AI complète sur les tests de drift avant d'écrire ce module — en particulier les différences entre `DataDriftPreset`, `DataQualityPreset`, et les tests statistiques disponibles (KS test, PSI, Wasserstein distance). Choisir le bon test selon le type de feature (numérique vs catégorielle).

**`monitoring/drift_detector.py`** : classe `DriftDetector` avec deux méthodes principales.

```python
class DriftDetector:

    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        window_start: datetime,
        window_end: datetime
    ) -> DriftReport:
        """
        Compare la distribution des features entre reference et current.
        Retourne un DriftReport avec drift_score global, liste des features
        qui dérivent, sévérité (ok/warning/critical), et rapport Evidently
        complet en JSON pour stockage.
        """

    def detect_concept_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        predictions_ref: np.ndarray,
        predictions_current: np.ndarray,
        window_start: datetime,
        window_end: datetime
    ) -> DriftReport:
        """
        Détecte si la relation features → prédictions a changé.
        Utilise TargetDriftPreset d'Evidently.
        """
```

Définir un dataclass `DriftReport` avec tous les champs nécessaires pour le stockage et l'affichage.

**Logique de sévérité** : les seuils viennent de `core/config.py`. Le drift_score global est la proportion de features qui dérivent significativement. En dessous du seuil warning : OK (vert). Entre warning et critical : Warning (orange). Au dessus de critical : Critical (rouge).

**✅ Validation** : tester avec des données identiques (drift_score ≈ 0), avec le scénario Gradual Drift (drift_score croissant), et avec le scénario Sudden Drift (drift_score élevé). Les trois cas doivent produire des sévérités différentes.

---

## ✅ Étape 4 — Suivi de performance (TERMINÉE)

**`monitoring/performance_tracker.py`** : classe `PerformanceTracker`.

```python
class PerformanceTracker:

    def compute_metrics(
        self,
        predictions_df: pd.DataFrame,
        window_start: datetime,
        window_end: datetime
    ) -> PerformanceMetrics | None:
        """
        Calcule accuracy, precision, recall, F1, ROC-AUC sur la fenêtre.
        Retourne None si moins de 30 samples avec labels réels disponibles.
        En production, les labels arrivent avec du retard — ce cas est normal.
        """

    def compute_performance_degradation(
        self,
        current_metrics: PerformanceMetrics,
        reference_metrics: PerformanceMetrics
    ) -> dict[str, float]:
        """
        Calcule le % de dégradation par métrique vs les métriques de référence.
        """
```

**Cas important** : en production réelle, les labels vrais (actual_label) n'arrivent pas immédiatement. Le tracker doit fonctionner en mode "pas de labels disponibles" (monitoring du drift uniquement) et en mode "labels disponibles" (monitoring de performance complète). Ce détail montre une compréhension des contraintes réelles de production.

**✅ Validation** : tester avec un DataFrame où 50% des labels sont NULL. Les métriques ne sont calculées que sur les lignes avec labels. `compute_performance_degradation` retourne des deltas corrects.

---

## ✅ Étape 5 — Système d'alertes (TERMINÉE)

**`monitoring/alerting.py`** : classe `AlertManager`.

```python
class AlertManager:

    def evaluate_alerts(
        self,
        drift_report: DriftReport,
        performance_metrics: PerformanceMetrics | None
    ) -> list[Alert]:
        """
        Évalue les conditions d'alerte et retourne la liste des alertes actives.
        """

    def get_active_alerts(self, last_n_hours: int = 24) -> list[Alert]:
        """
        Récupère les alertes actives depuis SQLite.
        """
```

Dataclass `Alert` : `type` (data_drift / concept_drift / performance_degradation), `severity` (warning / critical), `message` humainement lisible, `timestamp`, `affected_features`.

**Règles d'alerte :**
- Data drift warning / critical selon les seuils de config
- Performance degradation : F1 dégradé de plus de 10% vs référence
- Missing labels : plus de 48h sans labels réels (impossible de monitorer la performance)

Les messages d'alerte doivent être lisibles par un non-technicien — ils seront lus par des business analysts et des managers.

**✅ Validation** : simuler chaque condition d'alerte individuellement. Vérifier que les messages sont clairs sans jargon technique.

---

## ✅ Étape 6 — Pipeline de monitoring automatique (TERMINÉE)

**`run_monitoring.py`** : script autonome qui simule le monitoring quotidien. Accepte un argument `--scenario` (stable / gradual_drift / sudden_drift).

Séquence d'exécution :
1. Générer un batch de données de production pour la date courante via le simulateur
2. Faire les prédictions avec le modèle de référence
3. Logger les prédictions dans SQLite
4. Récupérer les données des 7 derniers jours depuis SQLite
5. Lancer la détection de drift (data + concept)
6. Calculer les métriques de performance si des labels sont disponibles
7. Évaluer les alertes
8. Sauvegarder les rapports dans SQLite
9. Logger un résumé clair avec `loguru`

Ce script est aussi utilisé pour pré-remplir la base SQLite avant le déploiement : lancer 60 fois en boucle (30 jours stable + 30 jours gradual_drift) pour générer un historique riche.

**✅ Validation** : lancer 30 fois avec `--scenario gradual_drift`. Vérifier dans SQLite que les drift_scores augmentent progressivement. Les alertes doivent se déclencher à partir d'un certain jour.

---

## Étape 7 — Dashboard Streamlit

**`app.py`** — quatre sections dans la sidebar.

**Section 1 — Overview**

Métriques clés avec `st.metric` : statut global (OK / Warning / Critical) avec couleur, drift score actuel, dernière mise à jour, nombre d'alertes actives.

Timeline interactive Plotly sur 30 jours : évolution du drift score avec lignes horizontales pour les seuils. Points colorés selon la sévérité. Cliquer sur un point affiche le détail du rapport de ce jour.

**Section 2 — Data Drift**

Deux colonnes : liste des features avec drift score individuel (barre de progression colorée), et détail de la feature sélectionnée — superposition des distributions reference vs current (histogramme pour les numériques, barres pour les catégorielles). Sélecteur de fenêtre temporelle : 7 / 14 / 30 jours.

**Section 3 — Model Performance**

Graphiques temporels Plotly de accuracy, F1, AUC. Ligne de référence en pointillés. Zone grisée quand les labels ne sont pas disponibles. Tableau des dernières prédictions avec timestamp, features principales, prédiction, probabilité, label réel si disponible.

**Section 4 — Alerts**

Liste des alertes actives avec badge de sévérité coloré, message, timestamp, features concernées. Historique des alertes sur 30 jours. Bouton "Acknowledge" qui marque une alerte comme vue dans SQLite.

**Contrôle de scénario (pour la démo)**

Dans la sidebar : sélecteur de scénario (Stable / Gradual Drift / Sudden Drift) et bouton "Simulate next day" qui appelle `run_monitoring.py`. Permet aux recruteurs de voir le système réagir en temps réel pendant la démo — c'est la fonctionnalité la plus importante pour l'entretien.

**`@st.cache_data`** sur toutes les fonctions qui lisent SQLite avec `ttl=300`.

**Gestion des erreurs** : si SQLite est vide, afficher un message d'onboarding avec les instructions pour lancer `run_monitoring.py`. Jamais de stack trace visible.

**✅ Validation** : dashboard fonctionnel avec 60 jours de données. Les trois scénarios produisent des visualisations distinctes. Le bouton "Simulate next day" met à jour les graphiques. Aucune stack trace visible.

---

## Étape 8 — Tests unitaires

Tests avec `pytest` + `unittest.mock`. Pas de vrais appels à la base de données dans les tests unitaires — mocker `storage.py`.

**`tests/test_drift_detector.py`** :
- `detect_data_drift` retourne `severity="ok"` avec des données identiques
- Sévérité `"critical"` avec des distributions sans overlap
- Comportement avec un DataFrame vide
- Features catégorielles traitées différemment des numériques

**`tests/test_performance_tracker.py`** :
- `compute_metrics` retourne `None` avec moins de 30 samples labellisés
- `compute_performance_degradation` retourne des deltas corrects
- Comportement quand tous les labels sont NULL

**`tests/test_simulator.py`** :
- Les trois scénarios produisent des distributions statistiquement différentes (test KS)
- `generate_production_batch` retourne le bon nombre de samples
- Reproductibilité avec un `random_state` fixé

**✅ Validation** : `pytest tests/ -v` → tous verts.

---

## Étape 9 — Déploiement

**Pré-remplir la base SQLite** : générer 60 jours de données (30 stable + 30 gradual_drift) et committer `monitoring.db` dans le repo. Le dashboard Streamlit Cloud doit s'ouvrir avec des données riches dès le premier chargement — critique pour l'impression lors d'une démo recruteur.

**README.md** (en anglais) : GIF de démonstration montrant le changement de scénario en temps réel. Section Key Technical Decisions avec au minimum : pourquoi Evidently plutôt qu'une implémentation custom des tests statistiques, pourquoi SQLite plutôt que PostgreSQL, et pourquoi la simulation de dérive plutôt qu'un vrai dataset.

**Streamlit Community Cloud** : pas de secrets nécessaires (pas d'API externe). Déploiement direct depuis GitHub. Vérifier que `monitoring.db` est bien dans le repo et pas dans `.gitignore`.

**✅ Validation** : URL Streamlit Cloud accessible. Dashboard s'ouvre avec 60 jours de données. Le changement de scénario fonctionne en production.

---

## Bonnes pratiques à appliquer dans tout le code

**Séparation des responsabilités** : `drift_detector.py` ne sait pas que Streamlit existe. `app.py` ne contient aucune logique de calcul — seulement des appels aux modules `monitoring/` et `data/`.

**Dataclasses typées** : `DriftReport`, `PerformanceMetrics`, `Alert` sont des dataclasses — pas des dictionnaires. Rend le code lisible et les erreurs détectables tôt.

**Simulation réaliste** : les trois scénarios doivent être visuellement et statistiquement distincts. Un recruteur technique regardera les graphiques — si la dérive est imperceptible, le projet ne démontre rien.

**Seuils configurables** : tous les seuils d'alerte dans `core/config.py`, jamais hardcodés dans la logique métier.

**Type hints et docstrings** partout sur les fonctions et classes publiques.

**`loguru`** pour tout le logging avec niveaux appropriés.

---

## Checklist finale

- [ ] Les trois scénarios de simulation sont visuellement distincts dans le dashboard
- [ ] Les alertes se déclenchent aux bons seuils
- [ ] `pytest tests/ -v` → tous verts
- [ ] `monitoring.db` pré-remplie avec 60 jours committée dans le repo
- [ ] README avec GIF de démonstration et section Key Technical Decisions
- [ ] URL Streamlit Cloud accessible et fonctionnelle
- [ ] Dashboard s'ouvre avec des données riches dès le premier chargement
