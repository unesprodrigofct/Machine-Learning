"""CLI script to train the XGBoost pipeline."""

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import CSVLoaderConfig, load_csv_dataset, load_titanic_for_xgboost
from src.core.base import ModelMetadata
from src.core.trainers import XGBoostConfig, XGBoostTrainer


def main() -> None:
    dataset_path = Path("data/xgboost_dataset.csv")
    target_column = "TARGET"

    if dataset_path.exists():
        df = load_csv_dataset(CSVLoaderConfig(path=dataset_path))
    else:
        df = load_titanic_for_xgboost(cache_dir=Path("data"))
    metadata = ModelMetadata(name="xgboost_model", artifact_path=Path("artifacts"))

    trainer = XGBoostTrainer(
        config=XGBoostConfig(),
        metadata=metadata,
    )
    auc, ks_stat = trainer.train(df, target_column)
    trainer.export()
    print(f"Training complete. ROC-AUC: {auc:.3f} | KS: {ks_stat:.3f}")


if __name__ == "__main__":
    main()
