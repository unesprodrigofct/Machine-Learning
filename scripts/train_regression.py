"""CLI script to train the regression pipeline."""

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.base import ModelMetadata
from src.core.trainers import LinearRegressionTrainer, RegressionConfig


def main() -> None:
    metadata = ModelMetadata(name="linear_regression", artifact_path=Path("artifacts"))
    trainer = LinearRegressionTrainer(RegressionConfig(test_size=0.2, random_state=42), metadata)
    mse, r2 = trainer.train()
    trainer.export()
    print(f"Training complete. MSE: {mse:.2f} | R2: {r2:.2f}")


if __name__ == "__main__":
    main()
