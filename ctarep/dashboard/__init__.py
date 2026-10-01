"""Dashboard rendering layer.

Figure creation (`figures.FigureFactory`) and HTML assembly
(`builder.DashboardBuilder`) are separate, so FigureFactory can be reused as is
if the dashboard moves to a server-based Dash/Streamlit app.
"""
from .builder import Card, DashboardBuilder, Tab
from .figures import FigureFactory
from .theme import DARK, LIGHT, THEMES, Theme

__all__ = ["DashboardBuilder", "Card", "Tab", "FigureFactory", "Theme", "LIGHT", "DARK", "THEMES"]
