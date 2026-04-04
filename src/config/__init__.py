"""
Configuration package.

Exports the singleton Settings instance and accessor function.
"""

from src.config.settings import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
