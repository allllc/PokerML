# PokerML - Machine Learning Pipeline for Poker Decision Making

A comprehensive machine learning pipeline for analyzing poker hand histories and training models to predict optimal poker decisions. Built for Databricks serverless compute.

## Overview

This project implements a full ML pipeline that:
- Parses poker hand history data
- Engineers features from player actions and game state
- Trains opponent modeling classifiers
- Builds profit prediction models
- Trains policy models on winning player strategies

## Pipeline Architecture

```
01_DataCollection → 02_FeatureEngineering → 03_OpponentModeling
                                                      ↓
04_AdvancedFeatures → 05_ProfitModel
                              ↓
06_PolicyFeatures → 07_LabelCreation → 08_PlayerPerformance → 09_PolicyTraining
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `00_DataValidation` | Validates data integrity across pipeline stages |
| `01_DataCollection` | Downloads and parses poker hand histories |
| `02_FeatureEngineering` | Creates player event features and labels |
| `03_OpponentModeling` | Trains opponent action classification models |
| `04_AdvancedFeatures` | Generates advanced statistical features |
| `05_ProfitModel` | Trains profit/EV prediction models |
| `06_PolicyFeatures` | Engineers features for policy training |
| `07_LabelCreation` | Creates action labels for policy models |
| `08_PlayerPerformance` | Identifies winning players (>= 5 BB/100) |
| `09_PolicyTraining` | Trains policy advisor on winning player actions |
| `PIPELINE` | Orchestrates full pipeline execution |

## Key Features

- **Opponent Modeling**: Classifies opponent tendencies per street (preflop, flop, turn, river)
- **Profit Prediction**: Estimates expected value of actions
- **Policy Learning**: Learns from top-performing players to recommend optimal actions
- **MLflow Integration**: Full experiment tracking with metrics, parameters, and artifacts

## Requirements

- Databricks workspace with serverless compute
- Python 3.x with:
  - PySpark
  - scikit-learn
  - MLflow
  - pandas
  - numpy

## Usage

### Running on Databricks

1. Import the notebooks to your Databricks workspace
2. Configure the `PIPELINE.ipynb` settings:
   - `DEBUG_MODE`: Set to `True` for testing with sampled data
   - `MAX_ROWS`: Maximum rows to process
   - `TRAINING_SAMPLE_SIZE`: Sample size for model training
3. Run `PIPELINE.ipynb` to execute the full pipeline, or run notebooks individually

### Pipeline Configuration

```python
DEBUG_MODE = False          # True = sample data, False = full data
SAMPLE_FRACTION = 0.01      # Sampling fraction (if DEBUG_MODE=True)
MAX_ROWS = 10000000         # Maximum rows to process
TRAINING_SAMPLE_SIZE = 300000  # Samples for sklearn training
```

## Models Trained

- **Opponent Models** (per street): Predict opponent action probabilities
- **Profit Models** (per street): Predict expected profit from actions
- **Policy Models** (per street): Recommend optimal actions based on winning player behavior

## Data Flow

1. Raw hand histories → Parsed CSV files
2. Parsed data → Feature-engineered datasets
3. Features → Trained models (saved as `.joblib`)
4. Models → MLflow experiment tracking

## License

MIT License

## Author

Leo Lwakabamba - llwakaba@asu.edu
