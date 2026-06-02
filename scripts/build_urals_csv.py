"""
Собирает data/urals.csv из дневного Brent (daily_raw_data.csv).

Официальный ряд Urals (Минэкономразвития / Argus) в открытом CSV недоступен,
поэтому используется оценка: средний Brent за месяц минус дисконт Urals к Brent.
Дисконты по годам/месяцам — ориентиры по публикациям Минэка и рынку (2014–2021 ~2–4 $,
с 2022 — шире). При появлении официального файла замените data/urals.csv вручную.
"""

from __future__ import annotations

import csv
from calendar import monthrange
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DAILY_PATH = DATA_DIR / "daily_raw_data.csv"
OUTPUT_PATH = DATA_DIR / "urals.csv"

START_YEAR = 2014
END_YEAR = 2024

# Дисконт Brent − Urals, USD/барр. (ключ YYYY-MM или YYYY)
DISCOUNT_USD: dict[str, float] = {
    "2014": 2.8,
    "2015": 2.5,
    "2016": 2.5,
    "2017": 2.6,
    "2018": 2.8,
    "2019": 3.0,
    "2020": 3.0,
    "2021": 3.2,
    "2022-01": 18.0,
    "2022-02": 22.0,
    "2022-03": 24.0,
    "2022-04": 28.0,
    "2022-05": 30.0,
    "2022-06": 28.0,
    "2022-07": 25.0,
    "2022-08": 22.0,
    "2022-09": 20.0,
    "2022-10": 22.0,
    "2022-11": 24.0,
    "2022-12": 26.0,
    "2023": 18.0,
    "2024": 15.0,
}


def month_end(year: int, month: int) -> date:
    return date(year, month, monthrange(year, month)[1])


def discount_for(year: int, month: int) -> float:
    key = f"{year}-{month:02d}"
    if key in DISCOUNT_USD:
        return DISCOUNT_USD[key]
    return DISCOUNT_USD[str(year)]


def main() -> None:
    if not DAILY_PATH.exists():
        raise FileNotFoundError(f"Нет файла {DAILY_PATH}. Сначала запустите src.py или скачайте дневные данные.")

    sums: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)

    with DAILY_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            day = date.fromisoformat(row["date"][:10])
            if day.year < START_YEAR or day.year > END_YEAR:
                continue
            key = (day.year, day.month)
            sums[key] += float(row["brent"])
            counts[key] += 1

    rows: list[tuple[str, float]] = []
    for year, month in sorted(sums.keys()):
        brent_avg = sums[(year, month)] / counts[(year, month)]
        urals = round(brent_avg - discount_for(year, month), 4)
        rows.append((month_end(year, month).isoformat(), urals))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "urals"])
        writer.writerows(rows)

    print(f"Записано {len(rows)} месяцев в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
