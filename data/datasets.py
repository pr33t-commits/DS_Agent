"""Shared checkout-relative input locations; raw CSVs remain in data/."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "DataCoSupplyChainDataset.csv"
DEFAULT_COLUMNS = PROJECT_ROOT / "data" / "DescriptionDataCoSupplyChain.csv"
