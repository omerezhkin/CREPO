"""
Схема четырёх классов признаков для слайда «Генерация признаков».
Сохраняет figures/feature_generation_slide.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
DATA_PATH = ROOT / "data" / "daily_features.csv"

# Палитра под светлый слайд
COLOR_PRICE = "#1f4e79"
COLOR_MA = "#6b9bc3"
COLOR_VOL = "#c45c3e"
COLOR_RETURN = "#3d7a5a"
COLOR_LAG = "#7a6b9b"
TEXT = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e8e8e8"


def _style_axes(ax, title, subtitle):
    ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT, loc="left", pad=8)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def plot_feature_generation_slide(
    output_path=None,
    start="2019-06-01",
    end="2020-06-30",
    oil="brent",
):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    window = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if len(window) < 60:
        window = df.iloc[-180:].copy()

    price = window[oil]
    ret = window[f"{oil}_return"]
    ma7 = window[f"{oil}_ma7"]
    vol7 = window[f"{oil}_vol7"]
    lag30 = window[f"{oil}_lag30"]
    dates = window["date"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(6.2, 7.4), dpi=200)
    fig.subplots_adjust(hspace=0.42, wspace=0.28, left=0.1, right=0.96, top=0.94, bottom=0.07)

    # 1 — мгновенные изменения (доходность)
    ax = axes[0, 0]
    colors = np.where(ret >= 0, COLOR_RETURN, COLOR_VOL)
    ax.bar(dates, ret * 100, width=2.5, color=colors, alpha=0.85, linewidth=0)
    ax.axhline(0, color=MUTED, linewidth=0.7)
    _style_axes(ax, "1 · Мгновенные изменения", f"{oil}_return, % в день")
    ax.set_ylabel("%", fontsize=7, color=MUTED)

    # 2 — тренд (скользящие средние)
    ax = axes[0, 1]
    ax.plot(dates, price, color=COLOR_PRICE, linewidth=1.4, label="Цена")
    ax.plot(dates, ma7, color=COLOR_MA, linewidth=1.2, linestyle="--", label="MA7")
    ax.plot(dates, window[f"{oil}_ma30"], color=COLOR_MA, linewidth=1.2, linestyle=":", label="MA30")
    _style_axes(ax, "2 · Трендовые компоненты", f"{oil}_ma7, {oil}_ma30")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", labelcolor=MUTED)

    # 3 — неопределённость (волатильность)
    ax = axes[1, 0]
    ax.fill_between(dates, 0, vol7 * 100, color=COLOR_VOL, alpha=0.35)
    ax.plot(dates, vol7 * 100, color=COLOR_VOL, linewidth=1.2)
    _style_axes(ax, "3 · Уровень неопределённости", f"{oil}_vol7, скользящее σ доходности")
    ax.set_ylabel("σ, %", fontsize=7, color=MUTED)

    # 4 — лаги (запаздывание)
    ax = axes[1, 1]
    ax.plot(dates, price, color=COLOR_PRICE, linewidth=1.3, label="Сегодня")
    ax.plot(dates, lag30, color=COLOR_LAG, linewidth=1.2, linestyle="--", label="Лаг 30 дн.")
    _style_axes(ax, "4 · Запаздывающие эффекты", f"{oil}_lag30 … {oil}_lag1")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", labelcolor=MUTED)

    # Подпись внизу
    fig.text(
        0.5,
        0.01,
        "Brent и WTI · окна 7 и 30 дней · лаги 1, 3, 7, 14, 30",
        ha="center",
        fontsize=7.5,
        color=MUTED,
    )

    badges = [
        (0.02, 0.98, "22 признака", COLOR_PRICE),
        (0.18, 0.98, "×2 марки", COLOR_MA),
    ]
    for x, y, text, color in badges:
        fig.text(x, y, text, fontsize=7, color=color, fontweight="bold", va="top", ha="left")

    if output_path is None:
        output_path = FIGURES_DIR / "feature_generation_slide.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {output_path}")
    return output_path


if __name__ == "__main__":
    plot_feature_generation_slide()
