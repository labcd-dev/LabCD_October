"""Shared LabCD logo assets for the Streamlit workspace."""

from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "assets"
ICON_PATH = ASSET_DIR / "labcd_icon.svg"
LOGO_PATH = ASSET_DIR / "labcd_logo.svg"
ICON_MARKUP = ICON_PATH.read_text(encoding="utf-8")
LOGO_MARKUP = LOGO_PATH.read_text(encoding="utf-8")
