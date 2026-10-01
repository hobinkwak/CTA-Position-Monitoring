"""Dashboard colour and typography tokens.

Categorical series use a 12-slot palette (the first eight are blue, orange, aqua, yellow,
magenta, green, violet, red; slots 9-12 add teal, brown, slate, lime) stepped separately for the light and dark surfaces. The four
sectors take slots 1 to 4 in a fixed order, so a sector keeps its colour when other
series are filtered out. Signed values (long/short) use a blue-red diverging pair
with a neutral gray midpoint. To re-skin the dashboard, edit `series`, `pos_pole`,
`neg_pole` and `sequential` in `LIGHT` and `DARK` below; nothing else needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Full token set for one mode (light or dark)."""

    name: str

    # surfaces & ink
    surface: str
    page: str
    text_primary: str
    text_secondary: str
    muted: str
    gridline: str
    baseline: str
    border: str

    # categorical slots (the four sectors take slots 1-4)
    series: tuple[str, ...]

    # diverging (long / short positions)
    pos_pole: str  # blue = long
    neg_pole: str  # red  = short
    mid: str  # neutral gray

    # sequential (magnitude)
    sequential: tuple[str, ...]

    # status
    good: str = "#0ca30c"
    warning: str = "#fab219"
    serious: str = "#ec835a"
    critical: str = "#d03b3b"

    font: str = 'system-ui, -apple-system, "Segoe UI", sans-serif'

    # ------------------------------------------------------------------ #

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"

    def sector_colors(self, sectors: list[str]) -> dict[str, str]:
        """Sector -> colour. Slot order is fixed (colour follows the entity, not its rank),
        so filtering the series never changes the colour of a surviving sector.
        """
        return {sector: self.series[i % len(self.series)] for i, sector in enumerate(sectors)}

    def sign_color(self, value: float) -> str:
        if value > 0:
            return self.pos_pole
        if value < 0:
            return self.neg_pole
        return self.mid

    def scenario_colors(self, sigmas: list[float]) -> list[str]:
        """Diverging colours for the five scenarios.

        Scenarios are an ordered polarity (strong down to strong up), so a diverging scale
        is used instead of categorical colours. The neutral (flat) scenario is gray, drawn
        with the baseline tone because it is used as a bar fill and must stay visible on the
        surface.
        """
        peak = max((abs(s) for s in sigmas), default=1.0) or 1.0
        out = []
        for sigma in sigmas:
            if sigma == 0:
                out.append(self.baseline)
                continue
            pole = self.pos_pole if sigma > 0 else self.neg_pole
            # weaker scenarios fade from the pole toward the surface (same hue, lighter)
            out.append(_mix(pole, self.surface, 1.0 - abs(sigma) / peak, max_fade=0.55))
        return out

    def diverging_scale(self) -> list[list]:
        """Plotly diverging colorscale (0 = negative, 0.5 = neutral, 1 = positive)."""
        return [
            [0.0, self.neg_pole],
            [0.25, _mix(self.neg_pole, self.surface, 0.45)],
            [0.5, self.mid],
            [0.75, _mix(self.pos_pole, self.surface, 0.45)],
            [1.0, self.pos_pole],
        ]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _mix(color: str, toward: str, ratio: float, max_fade: float = 1.0) -> str:
    """Mix `color` toward `toward` by `ratio` (0 = original colour, 1 = toward)."""
    ratio = max(0.0, min(max_fade, ratio))
    a, b = _hex_to_rgb(color), _hex_to_rgb(toward)
    mixed = tuple(int(round(a[i] * (1 - ratio) + b[i] * ratio)) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


#: Slots 1-4 = sector totals (equities / bonds / currencies / commodities); others follow.
LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    gridline="#e1e0d9",
    baseline="#c3c2b7",
    border="rgba(11,11,11,0.10)",
    series=(
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
        "#0e9aa7", "#8b5a2b", "#5b6b7a", "#8fb300",
    ),
    pos_pole="#2a78d6",
    neg_pole="#e34948",
    mid="#f0efec",
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    gridline="#2c2c2a",
    baseline="#383835",
    border="rgba(255,255,255,0.10)",
    series=(
        "#3987e5", "#d95926", "#199e70", "#c98500",
        "#d55181", "#008300", "#9085e9", "#e66767",
        "#2bb4c0", "#a9774a", "#8a9aa8", "#a6cc1f",
    ),
    pos_pole="#3987e5",
    neg_pole="#e66767",
    mid="#383835",
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"),
)

THEMES: dict[str, Theme] = {"light": LIGHT, "dark": DARK}


__all__ = ["Theme", "LIGHT", "DARK", "THEMES"]
