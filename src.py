import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("data") / ".matplotlib_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning as StatsmodelsConvergenceWarning


START_DATE = "2014-01-01"
END_DATE = "2024-12-31"
DATA_DIR = Path("data")
FIGURES_DIR = Path("figures")

OIL_COLUMNS = ["brent", "wti"]
LAGS = [1, 3, 7, 14, 30]
ROLLING_WINDOWS = [7, 30]
TEST_SIZE = 0.2


def _download_close(ticker, start_date, end_date):
    end_plus_one = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(
        ticker,
        start=start_date,
        end=end_plus_one,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"Не удалось загрузить данные для {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            close = data[("Close", ticker)]
        elif ("Adj Close", ticker) in data.columns:
            close = data[("Adj Close", ticker)]
        else:
            close_columns = [col for col in data.columns if col[0] in ["Close", "Adj Close"]]
            if not close_columns:
                raise ValueError(f"Не найден столбец Close для {ticker}")
            close = data[close_columns[0]]
    else:
        if "Close" in data.columns:
            close = data["Close"]
        elif "Adj Close" in data.columns:
            close = data["Adj Close"]
        else:
            raise ValueError(f"Не найден столбец Close для {ticker}")

    close = close.dropna()
    if close.empty:
        raise ValueError(f"Пустой ряд цен для {ticker}")
    return close


def download_daily_data(start_date=START_DATE, end_date=END_DATE):
    brent = _download_close("BZ=F", start_date, end_date).rename("brent")
    wti = _download_close("CL=F", start_date, end_date).rename("wti")

    usdrub = None
    last_error = None
    for ticker in ["RUB=X", "USDRUB=X"]:
        try:
            usdrub = _download_close(ticker, start_date, end_date).rename("usdrub")
            break
        except Exception as error:
            last_error = error

    if usdrub is None:
        raise RuntimeError("Не удалось загрузить USD/RUB через RUB=X или USDRUB=X") from last_error

    df = pd.concat([usdrub, brent, wti], axis=1, sort=False).reset_index()
    df = df.rename(columns={"Date": "date", "Datetime": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values("date")


def apply_raw_data_overrides(df):
    """Подмешивает правки из data/raw_data.csv (date, brent, usdrub) в дневной ряд."""
    raw_path = DATA_DIR / "raw_data.csv"
    if not raw_path.exists():
        return df

    raw = pd.read_csv(raw_path, parse_dates=["date"])
    if not {"date", "brent", "usdrub"}.issubset(raw.columns):
        return df

    raw = raw[["date", "brent", "usdrub"]].copy()
    raw["brent"] = pd.to_numeric(raw["brent"], errors="coerce")
    raw["usdrub"] = pd.to_numeric(raw["usdrub"], errors="coerce")

    merged = df.merge(raw, on="date", how="left", suffixes=("", "_raw"))
    for col in ("brent", "usdrub"):
        raw_col = f"{col}_raw"
        if raw_col in merged.columns:
            merged[col] = merged[raw_col].combine_first(merged[col])
            merged = merged.drop(columns=[raw_col])
    return merged


def load_or_download_daily_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = DATA_DIR / "daily_raw_data.csv"
    old_path = DATA_DIR / "raw_data.csv"

    if daily_path.exists():
        df = pd.read_csv(daily_path, parse_dates=["date"])
        return apply_raw_data_overrides(df)

    try:
        df = download_daily_data()
        df = apply_raw_data_overrides(df)
        df.to_csv(daily_path, index=False)
        return df
    except Exception as error:
        if daily_path.exists():
            print(f"Не удалось скачать новые дневные данные, используем {daily_path}: {error}")
            df = pd.read_csv(daily_path, parse_dates=["date"])
            return apply_raw_data_overrides(df)
        if old_path.exists():
            old_df = pd.read_csv(old_path, parse_dates=["date"])
            if {"date", "usdrub", "brent", "wti"}.issubset(old_df.columns):
                print(f"Используем сохранённый {old_path}: {error}")
                return old_df[["date", "usdrub", "brent", "wti"]]
            if {"date", "usdrub", "brent"}.issubset(old_df.columns):
                print(f"Используем сохранённый {old_path} (без WTI): {error}")
                old_df["wti"] = pd.NA
                return apply_raw_data_overrides(old_df[["date", "usdrub", "brent", "wti"]])
        raise


def download_dubai_monthly():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=POILDUBUSDM"
    df = pd.read_csv(url)
    df = df.rename(columns={"observation_date": "date", "POILDUBUSDM": "dubai"})
    df["date"] = pd.to_datetime(df["date"])
    df["dubai"] = pd.to_numeric(df["dubai"], errors="coerce")
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    return df.dropna().sort_values("date")


def load_or_download_dubai_monthly():
    path = DATA_DIR / "dubai_monthly.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    try:
        df = download_dubai_monthly()
        df.to_csv(path, index=False)
        return df
    except Exception as error:
        if path.exists():
            print(f"Не удалось скачать Dubai из FRED, используем {path}: {error}")
            return pd.read_csv(path, parse_dates=["date"])
        print(f"Dubai не загружен, месячный блок будет без него: {error}")
        return pd.DataFrame(columns=["date", "dubai"])


def load_urals_if_available():
    path = DATA_DIR / "urals.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "urals"]), "Файл data/urals.csv не найден, поэтому Urals не входит в основную модель."

    df = pd.read_csv(path)
    if not {"date", "urals"}.issubset(df.columns):
        return pd.DataFrame(columns=["date", "urals"]), "В data/urals.csv нет столбцов date и urals."

    df["date"] = pd.to_datetime(df["date"])
    df["urals"] = pd.to_numeric(df["urals"], errors="coerce")
    df = df.dropna().sort_values("date")
    if len(df) < 100:
        return df, "Данных по Urals слишком мало для основной модели, используем только как справочный ряд."
    return df, "Urals загружен из локального data/urals.csv и добавлен в дополнительный месячный анализ."


def prepare_daily_data(raw_df):
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df[["date", "usdrub", "brent", "wti"]]
    df[["usdrub", "brent", "wti"]] = df[["usdrub", "brent", "wti"]].apply(pd.to_numeric, errors="coerce")

    df = df.set_index("date").sort_index()
    df = df.ffill().dropna().reset_index()
    return df


def add_daily_features(df):
    features = df.copy()
    features = features.sort_values("date")

    for col in ["usdrub"] + OIL_COLUMNS:
        features[f"{col}_return"] = features[col].pct_change()

    for col in OIL_COLUMNS:
        for window in ROLLING_WINDOWS:
            features[f"{col}_ma{window}"] = features[col].rolling(window).mean()
            features[f"{col}_vol{window}"] = features[f"{col}_return"].rolling(window).std()
        for lag in LAGS:
            features[f"{col}_lag{lag}"] = features[col].shift(lag)

    features = features.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return features


def make_monthly_benchmarks(daily_df, dubai_df, urals_df):
    monthly = daily_df.copy()
    monthly["date"] = pd.to_datetime(monthly["date"])
    try:
        monthly = monthly.set_index("date")[["usdrub", "brent", "wti"]].resample("ME").last().reset_index()
    except ValueError:
        monthly = monthly.set_index("date")[["usdrub", "brent", "wti"]].resample("M").last().reset_index()

    if not dubai_df.empty:
        dubai = dubai_df.copy()
        dubai["date"] = pd.to_datetime(dubai["date"]).dt.to_period("M").dt.to_timestamp("M")
        monthly = monthly.merge(dubai, on="date", how="left")

    if not urals_df.empty:
        urals = urals_df.copy()
        urals["date"] = pd.to_datetime(urals["date"]).dt.to_period("M").dt.to_timestamp("M")
        urals = urals.groupby("date", as_index=False)["urals"].last()
        monthly = monthly.merge(urals, on="date", how="left")

    return monthly.dropna(subset=["usdrub", "brent", "wti"])


def describe_data(df):
    numeric_cols = [col for col in df.columns if col != "date"]
    stats = df[numeric_cols].describe().T
    stats["missing"] = df[numeric_cols].isna().sum()
    return stats.reset_index().rename(columns={"index": "series"})


def calculate_correlations(features_df):
    cols = ["usdrub", "brent", "wti", "usdrub_return", "brent_return", "wti_return"]
    return features_df[cols].corr()


def cross_correlation(features_df, oil_col, max_lag=30):
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        shifted_oil = features_df[f"{oil_col}_return"].shift(lag)
        corr = shifted_oil.corr(features_df["usdrub_return"])
        rows.append({"indicator": oil_col, "lag": lag, "correlation": corr})
    result = pd.DataFrame(rows)
    idx = result["correlation"].abs().idxmax()
    best = result.loc[idx].to_dict()
    return result, best


def get_model_features(features_df):
    blocked = {"date", "usdrub", "usdrub_return"}
    return [col for col in features_df.columns if col not in blocked]


def split_train_test(features_df, feature_cols, target_col="usdrub", test_size=TEST_SIZE):
    df = features_df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    x_train = train[feature_cols]
    y_train = train[target_col]
    x_test = test[feature_cols]
    y_test = test[target_col]
    return train, test, x_train, x_test, y_train, y_test


def regression_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_ols_model(x_train, y_train):
    x_train_const = sm.add_constant(x_train)
    return sm.OLS(y_train, x_train_const).fit()


def run_single_factor_models(features_df):
    rows = []
    for oil in OIL_COLUMNS:
        feature_cols = [oil, f"{oil}_return", f"{oil}_ma30", f"{oil}_vol30", f"{oil}_lag30"]
        train, test, x_train, x_test, y_train, y_test = split_train_test(features_df, feature_cols)
        model = run_ols_model(x_train, y_train)
        pred = model.predict(sm.add_constant(x_test, has_constant="add"))
        metrics = regression_metrics(y_test, pred)
        rows.append(
            {
                "indicator": oil,
                "n_features": len(feature_cols),
                "main_beta": float(model.params[oil]),
                "main_p_value": float(model.pvalues[oil]),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def run_time_series_cv(features_df, feature_cols):
    df = features_df.sort_values("date").reset_index(drop=True)
    x = df[feature_cols]
    y = df["usdrub"]
    splitter = TimeSeriesSplit(n_splits=5)

    models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.05, max_iter=50000, tol=1e-3))]),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.04, max_depth=3, random_state=42),
    }

    rows = []
    for model_name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x), start=1):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SklearnConvergenceWarning)
                model.fit(x_train, y_train)
            pred = model.predict(x_test)
            rows.append({"model": model_name, "fold": fold, **regression_metrics(y_test, pred)})

    return pd.DataFrame(rows)


def run_arimax_model(train, test, feature_cols):
    exog_train = train[feature_cols]
    exog_test = test[feature_cols]
    model = SARIMAX(
        train["usdrub"],
        exog=exog_train,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StatsmodelsConvergenceWarning)
        fitted = model.fit(disp=False, maxiter=150)
    forecast = fitted.forecast(steps=len(test), exog=exog_test)
    return fitted, forecast


def run_models(features_df):
    feature_cols = get_model_features(features_df)
    train, test, x_train, x_test, y_train, y_test = split_train_test(features_df, feature_cols)

    results = []
    predictions = test[["date", "usdrub"]].copy()
    feature_importance_parts = []

    ols_model = run_ols_model(x_train, y_train)
    ols_pred = ols_model.predict(sm.add_constant(x_test, has_constant="add"))
    predictions["OLS"] = ols_pred.values
    results.append({"model": "OLS", **regression_metrics(y_test, ols_pred)})

    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
    ridge.fit(x_train, y_train)
    ridge_pred = ridge.predict(x_test)
    predictions["Ridge"] = ridge_pred
    results.append({"model": "Ridge", **regression_metrics(y_test, ridge_pred)})
    ridge_coef = ridge.named_steps["model"].coef_
    feature_importance_parts.append(
        pd.DataFrame({"source": "Ridge_coef", "feature": feature_cols, "importance": ridge_coef})
    )

    lasso = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.05, max_iter=50000, tol=1e-3))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SklearnConvergenceWarning)
        lasso.fit(x_train, y_train)
    lasso_pred = lasso.predict(x_test)
    predictions["Lasso"] = lasso_pred
    results.append({"model": "Lasso", **regression_metrics(y_test, lasso_pred)})
    lasso_coef = lasso.named_steps["model"].coef_
    feature_importance_parts.append(
        pd.DataFrame({"source": "Lasso_coef", "feature": feature_cols, "importance": lasso_coef})
    )

    arimax_model, arimax_pred = run_arimax_model(train, test, feature_cols)
    predictions["ARIMAX"] = arimax_pred.values
    results.append({"model": "ARIMAX", **regression_metrics(y_test, arimax_pred)})

    rf = RandomForestRegressor(n_estimators=250, max_depth=8, random_state=42, n_jobs=1)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    predictions["Random Forest"] = rf_pred
    results.append({"model": "Random Forest", **regression_metrics(y_test, rf_pred)})
    feature_importance_parts.append(
        pd.DataFrame({"source": "RandomForest", "feature": feature_cols, "importance": rf.feature_importances_})
    )

    gb = GradientBoostingRegressor(n_estimators=250, learning_rate=0.04, max_depth=3, random_state=42)
    gb.fit(x_train, y_train)
    gb_pred = gb.predict(x_test)
    predictions["Gradient Boosting"] = gb_pred
    results.append({"model": "Gradient Boosting", **regression_metrics(y_test, gb_pred)})
    feature_importance_parts.append(
        pd.DataFrame({"source": "GradientBoosting", "feature": feature_cols, "importance": gb.feature_importances_})
    )

    try:
        from xgboost import XGBRegressor

        xgb_model = XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        )
        xgb_name = "XGBoost"
        xgb_model.fit(x_train, y_train)
        xgb_pred = xgb_model.predict(x_test)
        xgb_importance = xgb_model.feature_importances_
    except Exception:
        xgb_model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, random_state=42)
        xgb_name = "HistGradientBoosting"
        xgb_model.fit(x_train, y_train)
        xgb_pred = xgb_model.predict(x_test)
        perm = permutation_importance(
            xgb_model,
            x_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring="neg_mean_absolute_error",
        )
        xgb_importance = perm.importances_mean

    predictions[xgb_name] = xgb_pred
    results.append({"model": xgb_name, **regression_metrics(y_test, xgb_pred)})
    feature_importance_parts.append(
        pd.DataFrame({"source": xgb_name, "feature": feature_cols, "importance": xgb_importance})
    )

    model_results = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    feature_importance = pd.concat(feature_importance_parts, ignore_index=True)
    cv_results = run_time_series_cv(features_df, feature_cols)
    single_factor_results = run_single_factor_models(features_df)
    return {
        "feature_cols": feature_cols,
        "train": train,
        "test": test,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "ols_model": ols_model,
        "arimax_model": arimax_model,
        "model_results": model_results,
        "predictions": predictions,
        "feature_importance": feature_importance,
        "cv_results": cv_results,
        "single_factor_results": single_factor_results,
    }


def save_figures(daily_df, features_df, monthly_df, corr_matrix, lag_results, model_outputs):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily_df["date"], daily_df["brent"], label="Brent")
    ax.plot(daily_df["date"], daily_df["wti"], label="WTI")
    ax.set_title("Динамика нефтяных индикаторов")
    ax.set_xlabel("Дата")
    ax.set_ylabel("USD за баррель")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "oil_prices_daily.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily_df["date"], daily_df["usdrub"], color="#d62728")
    ax.set_title("Динамика USD/RUB")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Рублей за доллар")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "usdrub_daily.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(features_df["date"], features_df["brent_return"], label="Brent return", alpha=0.8)
    ax.plot(features_df["date"], features_df["wti_return"], label="WTI return", alpha=0.8)
    ax.plot(features_df["date"], features_df["usdrub_return"], label="USD/RUB return", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Дневные доходности")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Доходность")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "returns_daily.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Корреляционная матрица")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap_daily.png", dpi=160)
    plt.close(fig)

    for indicator, lag_df in lag_results.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(lag_df["lag"], lag_df["correlation"], marker="o", markersize=3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"Кросс-корреляция {indicator.upper()} и USD/RUB")
        ax.set_xlabel("Лаг нефтяной доходности, дней")
        ax.set_ylabel("Корреляция")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"cross_correlation_{indicator}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for col in [c for c in ["brent", "wti", "dubai", "urals"] if c in monthly_df.columns]:
        ax.plot(monthly_df["date"], monthly_df[col], label=col.upper())
    ax.set_title("Месячные нефтяные бенчмарки")
    ax.set_xlabel("Дата")
    ax.set_ylabel("USD за баррель")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_benchmarks.png", dpi=160)
    plt.close(fig)

    results = model_outputs["model_results"].sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results, x="RMSE", y="model", ax=ax, color="#4c78a8")
    ax.set_title("Сравнение моделей по RMSE")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Модель")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)

    rf_importance = model_outputs["feature_importance"]
    rf_importance = rf_importance[rf_importance["source"] == "RandomForest"].copy()
    rf_importance["abs_importance"] = rf_importance["importance"].abs()
    rf_importance = rf_importance.sort_values("abs_importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=rf_importance, x="importance", y="feature", ax=ax, color="#59a14f")
    ax.set_title("Важность признаков Random Forest")
    ax.set_xlabel("Важность")
    ax.set_ylabel("Признак")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance_rf.png", dpi=160)
    plt.close(fig)

    best_model = model_outputs["model_results"].iloc[0]["model"]
    predictions = model_outputs["predictions"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(predictions["date"], predictions["usdrub"], label="Факт", color="#1f77b4")
    ax.plot(predictions["date"], predictions[best_model], label=f"Прогноз: {best_model}", color="#d62728")
    ax.set_title("Факт и прогноз USD/RUB на test-периоде")
    ax.set_xlabel("Дата")
    ax.set_ylabel("USD/RUB")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_actual_vs_predicted.png", dpi=160)
    plt.close(fig)


def make_text_results(stats_df, corr_matrix, best_lags, model_outputs, urals_note):
    results = model_outputs["model_results"]
    best_model = results.iloc[0]
    rf_top = model_outputs["feature_importance"]
    rf_top = rf_top[rf_top["source"] == "RandomForest"].copy()
    rf_top["abs_importance"] = rf_top["importance"].abs()
    rf_top = rf_top.sort_values("abs_importance", ascending=False).head(5)

    lines = []
    lines.append("Краткие результаты")
    lines.append(f"Наблюдений в дневной выборке после построения признаков: {len(model_outputs['train']) + len(model_outputs['test'])}.")
    lines.append(f"Лучшая модель по RMSE: {best_model['model']} (RMSE = {best_model['RMSE']:.3f}).")
    lines.append(f"Корреляция Brent и USD/RUB: {corr_matrix.loc['brent', 'usdrub']:.3f}.")
    lines.append(f"Корреляция WTI и USD/RUB: {corr_matrix.loc['wti', 'usdrub']:.3f}.")
    for name, best in best_lags.items():
        lines.append(
            f"Максимальная по модулю кросс-корреляция для {name.upper()}: "
            f"lag={int(best['lag'])}, corr={best['correlation']:.3f}."
        )
    lines.append("Топ-5 признаков Random Forest:")
    for _, row in rf_top.iterrows():
        lines.append(f"- {row['feature']}: {row['importance']:.4f}")
    lines.append(urals_note)
    return "\n".join(lines)


def dataframe_to_markdown(df, float_digits=3):
    frame = df.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda value: f"{value:.{float_digits}f}")
    headers = list(frame.columns)
    rows = frame.astype(str).values.tolist()

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(daily_df, monthly_df, stats_df, corr_matrix, best_lags, model_outputs, urals_note):
    results = model_outputs["model_results"]
    best_model = results.iloc[0]
    ols_model = model_outputs["ols_model"]
    single_factor_results = model_outputs["single_factor_results"]
    cv_summary = (
        model_outputs["cv_results"]
        .groupby("model", as_index=False)[["MAE", "RMSE", "R2"]]
        .mean()
        .sort_values("RMSE")
    )

    ols_table = pd.DataFrame(
        {
            "coef": ols_model.params,
            "t": ols_model.tvalues,
            "p_value": ols_model.pvalues,
        }
    ).reset_index(names="feature")
    top_ols = ols_table[ols_table["feature"] != "const"].copy()
    top_ols["abs_t"] = top_ols["t"].abs()
    top_ols = top_ols.sort_values("abs_t", ascending=False).head(8)

    rf_top = model_outputs["feature_importance"]
    rf_top = rf_top[rf_top["source"] == "RandomForest"].copy()
    rf_top["abs_importance"] = rf_top["importance"].abs()
    rf_top = rf_top.sort_values("abs_importance", ascending=False).head(8)

    model_table = dataframe_to_markdown(results, float_digits=3)
    ols_md = dataframe_to_markdown(top_ols[["feature", "coef", "t", "p_value"]], float_digits=4)
    rf_md = dataframe_to_markdown(rf_top[["feature", "importance"]], float_digits=4)
    single_md = dataframe_to_markdown(single_factor_results, float_digits=3)
    cv_md = dataframe_to_markdown(cv_summary, float_digits=3)

    lag_text = "\n".join(
        [
            f"- {name.upper()}: максимальная по модулю связь при лаге {int(best['lag'])} дней, corr = {best['correlation']:.3f}."
            for name, best in best_lags.items()
        ]
    )

    dubai_text = "Dubai был загружен из FRED/IMF и использован в отдельном месячном блоке."
    if "dubai" not in monthly_df.columns or monthly_df["dubai"].dropna().empty:
        dubai_text = "Dubai не удалось загрузить, поэтому месячный блок построен без него."

    report = f"""# Влияние нефтяного рынка на курс рубля

Авторы: Мережкин Олег, Мамедов Сабухи, Заренков Пётр

## 1. Введение

В этой работе исследуется, как нефтяные индикаторы связаны с курсом USD/RUB. Логика простая: Россия получает значимую часть валютной выручки от экспорта сырья, поэтому нефтяные цены могут влиять на предложение валюты и на курс рубля. При росте нефтяных цен рубль теоретически может укрепляться, а USD/RUB снижаться.

На практике связь не является механической. На курс влияют санкции, валютные ограничения, политика Банка России, ожидания участников рынка и общий режим внешней торговли. Поэтому цель проекта не в точном предсказании курса, а в сравнении разных нефтяных факторов и моделей.

## 2. Описание данных

Основной дневной анализ построен по данным Yahoo Finance за период {START_DATE} - {END_DATE}. Используются:

- USD/RUB;
- Brent (`BZ=F`);
- WTI (`CL=F`).

{dubai_text} {urals_note}

После очистки и построения признаков в дневной выборке осталось {len(model_outputs['train']) + len(model_outputs['test'])} наблюдений. Последние 20% наблюдений используются как test-период, перемешивания нет.

## 3. Исследовательский анализ данных

Для каждого ряда построены графики динамики, описательные статистики и проверка пропусков. На графиках заметны кризисные периоды 2014-2015, 2020 и 2022 годов. Именно в такие моменты простая связь нефти и рубля может работать хуже.

Основные файлы с графиками находятся в папке `figures/`.

## 4. Корреляционный анализ

Корреляция Brent и USD/RUB: **{corr_matrix.loc['brent', 'usdrub']:.3f}**.

Корреляция WTI и USD/RUB: **{corr_matrix.loc['wti', 'usdrub']:.3f}**.

Корреляционная матрица показывает, что нефтяные бенчмарки сильно связаны между собой. Поэтому обычная линейная регрессия может сталкиваться с мультиколлинеарностью, а Ridge и Lasso здесь полезны как более устойчивые варианты.

## 5. Лаговый анализ

Кросс-корреляции считались для лагов от -30 до +30 дней. Положительный лаг означает, что нефтяная доходность берётся с задержкой относительно курса.

{lag_text}

Если максимальная связь находится не в нулевом лаге, это можно интерпретировать как признак задержанной реакции рынка. Но корреляция сама по себе не доказывает причинность.

## 6. Построение признаков

Для Brent и WTI были построены:

- текущие значения;
- дневные доходности;
- скользящие средние MA7 и MA30;
- волатильность за 7 и 30 дней;
- лаги 1, 3, 7, 14 и 30 дней.

Целевая переменная в сравнении моделей — текущий `usdrub`. Это объяснительная постановка: мы смотрим, насколько нефтяные признаки помогают объяснить уровень курса на той же дате.

## 7. Линейные модели

OLS показывает коэффициенты и их статистическую значимость. Наиболее заметные признаки по t-stat:

{ols_md}

Ridge используется, потому что Brent и WTI похожи и часто движутся вместе. Lasso помогает понять, какие признаки модель оставляет наиболее важными.

Отдельно были построены однофакторные модели для Brent и WTI:

{single_md}

## 8. ARIMAX

ARIMAX строится как модель временного ряда USD/RUB с нефтяными признаками как экзогенными переменными. Это основная эконометрическая модель в работе, потому что она учитывает и динамику самого курса, и внешние нефтяные факторы.

## 9. Модели машинного обучения

Также построены Random Forest, Gradient Boosting и XGBoost или, если XGBoost недоступен, HistGradientBoostingRegressor. Эти модели могут учитывать нелинейные зависимости, но их сложнее интерпретировать, чем OLS.

Топ признаков Random Forest:

{rf_md}

## 10. Сравнение результатов

Итоговая таблица качества:

{model_table}

Лучшая модель по RMSE: **{best_model['model']}**. Её RMSE равно **{best_model['RMSE']:.3f}**, MAE равно **{best_model['MAE']:.3f}**, R² равно **{best_model['R2']:.3f}**.

Среднее качество на TimeSeriesSplit:

{cv_md}

## 11. Экономическая интерпретация

Нефтяные признаки действительно дают информацию о курсе рубля, но не объясняют его полностью. Это ожидаемо: рубль зависит не только от нефтяной выручки, но и от санкций, движения капитала, бюджетного правила, валютных ограничений и политики ЦБ.

Использование нескольких нефтяных индикаторов полезно, но Brent и WTI очень похожи. Поэтому добавление WTI не всегда радикально улучшает прогноз, зато помогает проверить устойчивость результата.

## 12. Заключение

1. Brent и WTI связаны с USD/RUB, но эта связь не является полной моделью курса.
2. Лаговый анализ показывает, есть ли задержанная реакция курса на нефтяные движения.
3. Ridge и Lasso полезны из-за высокой похожести нефтяных бенчмарков.
4. ARIMAX удобен как эконометрическая модель, потому что соединяет временной ряд курса и внешние факторы.
5. Деревья и бустинг могут давать более гибкий прогноз, но их интерпретация менее прозрачна.
6. Для российской экономики вывод осторожный: нефть важна, но после 2014 и 2022 годов курс рубля определяется гораздо более широким набором факторов.
"""
    Path("report.md").write_text(report, encoding="utf-8")


def write_readme():
    text = """# Нефтяные индикаторы и курс рубля

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

```bash
pip install -r requirements.txt
python src.py
jupyter notebook notebook.ipynb
```

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
"""
    Path("README.md").write_text(text, encoding="utf-8")


def write_notebook():
    cells = [
        ("markdown", "# Влияние нефтяного рынка на курс рубля\n\nАвторы: Мережкин Олег, Мамедов Сабухи, Заренков Пётр"),
        ("markdown", "## 1. Введение\n\nВ проекте сравниваем Brent, WTI и отдельный месячный блок Dubai. Цель — понять, какие нефтяные признаки лучше объясняют USD/RUB."),
        (
            "code",
            "from pathlib import Path\n"
            "import os\n"
            "import sys\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'src.py').exists() and (ROOT.parent / 'src.py').exists():\n"
            "    ROOT = ROOT.parent\n"
            "os.chdir(ROOT)\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "import src\n\n"
            "if not Path('data/model_results.csv').exists():\n"
            "    src.main()",
        ),
        ("markdown", "## 2. Данные\n\nОсновная дневная выборка хранится в `data/daily_features.csv`. Месячные бенчмарки с Dubai — в `data/monthly_benchmarks.csv`."),
        ("code", "daily = pd.read_csv('data/daily_features.csv', parse_dates=['date'])\nmonthly = pd.read_csv('data/monthly_benchmarks.csv', parse_dates=['date'])\nresults = pd.read_csv('data/model_results.csv')\nimportance = pd.read_csv('data/feature_importance.csv')\n\ndisplay(daily.head())\nprint(daily.shape)\ndisplay(monthly.head())"),
        ("markdown", "## 3. EDA\n\nСначала смотрим динамику нефти и USD/RUB. На графиках должны быть заметны кризисные периоды и резкие движения курса."),
        ("code", "for name in ['oil_prices_daily.png', 'usdrub_daily.png', 'returns_daily.png']:\n    display(Image(filename=str(Path('figures') / name)))"),
        ("markdown", "## 4. Корреляции\n\nКорреляционная матрица показывает общую связь между уровнями и доходностями. Важно помнить, что корреляция не доказывает причинность."),
        ("code", "display(Image(filename='figures/correlation_heatmap_daily.png'))"),
        ("markdown", "## 5. Лаговый анализ\n\nЛаги от -30 до +30 дней помогают посмотреть, есть ли задержанная реакция курса на изменение нефтяных цен."),
        ("code", "display(Image(filename='figures/cross_correlation_brent.png'))\ndisplay(Image(filename='figures/cross_correlation_wti.png'))"),
        ("markdown", "## 6. Месячные бенчмарки\n\nDubai доступен как месячная серия FRED/IMF, поэтому он рассматривается отдельно от дневных моделей."),
        ("code", "display(Image(filename='figures/monthly_benchmarks.png'))"),
        ("markdown", "## 7. Сравнение моделей\n\nДля всех моделей считаются MAE, RMSE и R². Train/test делятся по времени, без перемешивания."),
        (
            "code",
            "single = pd.read_csv('data/single_factor_results.csv')\n"
            "cv = pd.read_csv('data/time_series_cv_results.csv')\n\n"
            "display(results)\n"
            "display(single)\n"
            "display(cv.groupby('model')[['MAE', 'RMSE', 'R2']].mean().sort_values('RMSE'))\n"
            "display(Image(filename='figures/model_comparison.png'))",
        ),
        ("markdown", "## 8. Важность признаков\n\nRandom Forest показывает, какие нефтяные признаки чаще используются для объяснения курса."),
        ("code", "display(importance.head(20))\ndisplay(Image(filename='figures/feature_importance_rf.png'))"),
        ("markdown", "## 9. Прогноз на test-периоде\n\nНа графике сравнивается фактический USD/RUB и прогноз лучшей модели по RMSE."),
        ("code", "display(Image(filename='figures/forecast_actual_vs_predicted.png'))"),
        ("markdown", "## 10. Выводы\n\nНефть влияет на рубль, но не объясняет курс полностью. Brent и WTI близки между собой, поэтому возникает мультиколлинеарность. После 2014 и 2022 годов связь курса и нефти становится менее стабильной из-за санкций, ограничений и изменения структуры валютного рынка."),
    ]

    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    for cell_type, source in cells:
        notebook["cells"].append(
            {
                "cell_type": cell_type,
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
            }
        )

    import json

    Path("notebook.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    daily_raw = load_or_download_daily_data()
    daily_clean = prepare_daily_data(daily_raw)
    daily_clean.to_csv(DATA_DIR / "daily_raw_data.csv", index=False)

    daily_features = add_daily_features(daily_clean)
    daily_features.to_csv(DATA_DIR / "daily_features.csv", index=False)

    dubai = load_or_download_dubai_monthly()
    urals, urals_note = load_urals_if_available()
    monthly = make_monthly_benchmarks(daily_clean, dubai, urals)
    monthly.to_csv(DATA_DIR / "monthly_benchmarks.csv", index=False)

    stats = describe_data(daily_clean)
    stats.to_csv(DATA_DIR / "descriptive_stats.csv", index=False)

    corr_matrix = calculate_correlations(daily_features)
    corr_matrix.to_csv(DATA_DIR / "correlation_matrix.csv")

    lag_results = {}
    best_lags = {}
    for oil in OIL_COLUMNS:
        lag_df, best = cross_correlation(daily_features, oil)
        lag_results[oil] = lag_df
        best_lags[oil] = best
        lag_df.to_csv(DATA_DIR / f"cross_correlation_{oil}.csv", index=False)

    model_outputs = run_models(daily_features)
    model_outputs["model_results"].to_csv(DATA_DIR / "model_results.csv", index=False)
    model_outputs["feature_importance"].to_csv(DATA_DIR / "feature_importance.csv", index=False)
    model_outputs["predictions"].to_csv(DATA_DIR / "model_predictions.csv", index=False)
    model_outputs["single_factor_results"].to_csv(DATA_DIR / "single_factor_results.csv", index=False)
    model_outputs["cv_results"].to_csv(DATA_DIR / "time_series_cv_results.csv", index=False)

    save_figures(daily_clean, daily_features, monthly, corr_matrix, lag_results, model_outputs)
    write_report(daily_clean, monthly, stats, corr_matrix, best_lags, model_outputs, urals_note)
    write_readme()
    write_notebook()

    print(make_text_results(stats, corr_matrix, best_lags, model_outputs, urals_note))


if __name__ == "__main__":
    main()
