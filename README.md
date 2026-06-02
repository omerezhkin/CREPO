# Нефтяные индикаторы и курс рубля

Расширенный учебный проект по риск-менеджменту и анализу временных рядов. Цель — сравнить, как разные нефтяные показатели связаны с USD/RUB, и проверить несколько простых моделей.

## Данные

- USD/RUB из Yahoo Finance.
- Brent: `BZ=F`.
- WTI: `CL=F`.
- Dubai: месячная серия FRED/IMF `POILDUBUSDM`.
- Urals: подключается только если есть локальный файл `data/urals.csv`.

## Что делает проект

- строит дневные нефтяные признаки;
- считает корреляции и лаги;
- строит OLS, Ridge, Lasso, ARIMAX;
- строит Random Forest, Gradient Boosting и XGBoost или HistGradientBoosting;
- сравнивает модели по MAE, RMSE и R²;
- сохраняет таблицы и графики.

## Запуск

На macOS используйте `python3` или виртуальное окружение `.venv`.

```bash
cd путь/к/рискмен_проект
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src.py
jupyter notebook presentations/notebook.ipynb
```

Ноутбук в `presentations/` сам переключает рабочую папку на корень проекта. После правок в `data/raw_data.csv` снова запустите `python src.py`.

## Основные файлы

```text
data/daily_raw_data.csv
data/daily_features.csv
data/monthly_benchmarks.csv
data/model_results.csv
data/feature_importance.csv
data/single_factor_results.csv
data/time_series_cv_results.csv
figures/
src.py
notebook.ipynb
report.md
```

## Ограничение

Основная модель объясняет USD/RUB на той же дате, а не прогнозирует следующий день. Это сделано специально, чтобы работа была понятной и хорошо интерпретируемой для учебного проекта.
