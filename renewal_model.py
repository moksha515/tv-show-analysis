"""
Season Odds: Show Renewability
==============================
Predict whether a show gets renewed or cancelled, using:
IMDb rating, number of votes, network, genre, and season 1 episode count.

Run from the folder that holds index.html:
    python renewal_model.py

It writes four charts into img/ (the ones index.html displays):
    img/renewal_by_genre.png     bar plot, renewal rate by genre
    img/renewal_by_network.png   bar plot, renewal rate by network
    img/roc.png                  ROC curve
    img/calibration.png          calibration plot
and a summary of the numbers into results/metrics.json.

DATA
----
Option A (recommended): the TMDB TV shows dataset from Kaggle
    kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows
Download the CSV into the data/ folder (any file name that contains "tmdb"
and "tv" works, for example data/TMDB_tv_dataset_v3.csv). The script builds
the `renewed` label itself. See build_from_tmdb() for the exact rules.

Option B: your own prepared file, data/tv_shows.csv, with the columns
    title, network, genre, imdb_rating, votes, s1_episodes, renewed
where `renewed` is 1 if the show got a season 2 and 0 if it was cancelled
after season 1 (the S1 to S2 decision).

You can also pass a path:  python renewal_model.py path/to/file.csv

If no data is found, the script builds a SYNTHETIC demo dataset so the whole
pipeline still runs. Charts made from demo data are for layout only and do
not say anything real about TV. Replace the data before you publish numbers.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path("data")
PREPARED_PATH = DATA_DIR / "tv_shows.csv"
IMG_DIR = Path("img")
RESULTS_DIR = Path("results")
RANDOM_STATE = 42

# Colors that match the website
BONE, SAGE, SAGE_DEEP = "#f4f3f0", "#7b9484", "#647c6d"
CHARCOAL, STONE = "#2a302d", "#6f766f"

NUMERIC = ["imdb_rating", "log_votes", "s1_episodes"]
CATEGORICAL = ["network", "genre"]


# ---------------------------------------------------------------------------
# 0. Data
# ---------------------------------------------------------------------------
def make_demo_data(n=1000):
    """Synthetic shows so the pipeline runs end to end. NOT real data."""
    rng = np.random.default_rng(RANDOM_STATE)
    networks = ["Netflix", "HBO", "Amazon", "FX", "AMC", "NBC", "ABC", "CBS"]
    genres = ["Drama", "Comedy", "Sci-Fi", "Crime", "Animation", "Reality"]
    net_effect = dict(zip(networks, [0.6, 0.5, 0.3, 0.2, 0.0, -0.3, -0.4, -0.3]))
    gen_effect = dict(zip(genres, [0.2, -0.1, -0.2, 0.3, 0.4, -0.4]))

    df = pd.DataFrame({
        "title": [f"Demo Show {i + 1}" for i in range(n)],
        "network": rng.choice(networks, n),
        "genre": rng.choice(genres, n),
        "imdb_rating": np.clip(rng.normal(7.3, 0.9, n), 4.5, 9.6).round(1),
        "s1_episodes": rng.choice([6, 8, 10, 13, 22], n, p=[.15, .25, .25, .2, .15]),
    })
    # Better-rated shows tend to collect more votes
    df["votes"] = np.exp(rng.normal(9 + 0.5 * (df["imdb_rating"] - 7.3), 1.2)).astype(int)

    logit = (0.2
             + 0.9 * (df["imdb_rating"] - 7.3)
             + 0.35 * (np.log1p(df["votes"]) - 9)
             - 0.04 * (df["s1_episodes"] - 10)
             + df["network"].map(net_effect)
             + df["genre"].map(gen_effect))
    p = 1 / (1 + np.exp(-logit))
    df["renewed"] = (rng.random(n) < p).astype(int)
    return df


# Settings for building the label from the TMDB file (change them if you like)
KEEP_TYPES = {"Scripted", "Reality"}  # drops miniseries, documentaries, talk shows, news
LABEL_MODE = "broad"    # how one-season shows are labeled: "broad" or "strict" (see build_from_tmdb)
MIN_VOTES = 50          # ignore shows that almost nobody rated
TOP_NETWORKS = 12       # smaller networks are grouped as "Other" so charts stay readable
TOP_GENRES = 8
MAX_S1_EPISODES = 30    # daily shows would otherwise stretch the scale


def first_item(value):
    """TMDB stores lists as comma separated text: 'Drama, Crime' -> 'Drama'."""
    if pd.isna(value):
        return np.nan
    first = str(value).split(",")[0].strip()
    return first or np.nan


def keep_top(series, n):
    top = series.value_counts().nlargest(n).index
    return series.where(series.isin(top), "Other")


def build_from_tmdb(path):
    """
    Turn the TMDB TV dataset into the table the model needs.

    The file has no "renewed" column, so we build one from what happened.
    Shows with 2 or more seasons are renewed = 1. For shows with exactly one
    season, the rule depends on LABEL_MODE:

      "strict"  renewed = 0 only if TMDB marks the show "Canceled"
      "broad"   renewed = 0 if TMDB marks it "Canceled" OR "Ended"

    Why two modes? Many cancelled shows are filed as "Ended" instead of
    "Canceled" (it varies a lot by network and country), so "strict" undercounts
    cancellations and pushes renewal rates toward 100%. "Broad" fixes that, but
    it also counts a few one-season shows that were meant to be one season as
    cancelled. Miniseries are already removed by KEEP_TYPES, which limits this.
    Neither is perfect, so the script prints both renewal rates.

    One-season shows that are still airing or planned are always dropped,
    because their fate is not known yet.

    Other choices made here:
      - imdb_rating is TMDB's vote_average (a stand-in for the IMDb rating)
      - s1_episodes is total episodes / total seasons. It is exact for the
        one-season shows and an average for the longer ones.
      - Only the first listed network and first listed genre are used.
    """
    wanted = {"name", "number_of_seasons", "number_of_episodes", "vote_count",
              "vote_average", "status", "genres", "networks", "type", "adult"}
    raw = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    required = {"name", "number_of_seasons", "number_of_episodes", "vote_count",
                "vote_average", "status", "genres", "networks"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}. "
                         f"Columns found: {list(raw.columns)}")
    print(f"Read {len(raw):,} shows from {path.name}")

    df = raw.copy()
    if "adult" in df:
        df = df[~df["adult"].fillna(False).astype(bool)]
    if "type" in df:
        df = df[df["type"].isin(KEEP_TYPES)]
    df = df[(df["vote_count"] >= MIN_VOTES) & (df["vote_average"] > 0)]
    df = df[df["number_of_seasons"] >= 1]

    # Build the renewed / cancelled label (see docstring)
    seasons = df["number_of_seasons"]
    status = df["status"].fillna("").str.lower()
    one_season = seasons == 1
    is_cancelled = status.isin(["canceled", "cancelled"])
    is_ended = status.eq("ended")

    print("\nStatus of one-season shows (this is what the label rule decides on):")
    print(df.loc[one_season, "status"].value_counts().to_string())

    n_renewed = int((seasons >= 2).sum())
    strict_zero = int((one_season & is_cancelled).sum())
    broad_zero = int((one_season & (is_cancelled | is_ended)).sum())
    print(f"Renewal rate under the strict rule: {n_renewed / (n_renewed + strict_zero):.1%}")
    print(f"Renewal rate under the broad rule:  {n_renewed / (n_renewed + broad_zero):.1%}")
    print(f"Using LABEL_MODE = '{LABEL_MODE}'\n")

    zero = one_season & (is_cancelled | is_ended) if LABEL_MODE == "broad" else one_season & is_cancelled
    df["renewed"] = np.select([seasons >= 2, zero], [1, 0], default=-1)
    df = df[df["renewed"] != -1]

    out = pd.DataFrame({
        "title": df["name"],
        "network": df["networks"].map(first_item),
        "genre": df["genres"].map(first_item),
        "imdb_rating": df["vote_average"],
        "votes": df["vote_count"],
        "s1_episodes": (df["number_of_episodes"] / df["number_of_seasons"])
                       .round().clip(1, MAX_S1_EPISODES),
        "renewed": df["renewed"].astype(int),
    }).dropna()
    out["network"] = keep_top(out["network"], TOP_NETWORKS)
    out["genre"] = keep_top(out["genre"], TOP_GENRES)

    n_ren, n_can = int((out["renewed"] == 1).sum()), int((out["renewed"] == 0).sum())
    print(f"Kept {len(out):,} shows: {n_ren:,} renewed, {n_can:,} cancelled after season 1")
    if n_can < 100:
        print("WARNING: very few cancelled shows. Try lowering MIN_VOTES.")
    # Heads up: vote counts cover the whole run of a show, so shows that lasted
    # longer have had more time to collect votes. That can make `votes` look more
    # predictive than it would be on a show's first season alone.
    return out.reset_index(drop=True)


def find_data_file():
    if PREPARED_PATH.exists():
        return PREPARED_PATH
    if DATA_DIR.exists():
        for f in sorted(DATA_DIR.glob("*.csv")):
            name = f.name.lower()
            if "tmdb" in name and "tv" in name:
                return f
    return None


def load_data(path=None):
    """Returns (dataframe, source_name)."""
    path = Path(path) if path else find_data_file()
    if path is None:
        print("WARNING: no data found in data/. Using SYNTHETIC demo data.")
        return make_demo_data(), "synthetic demo data"

    if "renewed" in pd.read_csv(path, nrows=0).columns:
        # Already prepared: title, network, genre, imdb_rating, votes, s1_episodes, renewed
        df = pd.read_csv(path)
        needed = {"network", "genre", "imdb_rating", "votes", "s1_episodes", "renewed"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        df = df.dropna(subset=list(needed))
        print(f"Loaded {len(df)} shows from {path}")
        return df, path.name
    return build_from_tmdb(path), path.name


# ---------------------------------------------------------------------------
# Plot styling (matches the website)
# ---------------------------------------------------------------------------
def set_style():
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": BONE, "axes.facecolor": BONE, "savefig.facecolor": BONE,
        "grid.color": "#d9dbd5", "axes.edgecolor": "#c3c7c0",
        "text.color": CHARCOAL, "axes.labelcolor": CHARCOAL,
        "xtick.color": STONE, "ytick.color": STONE,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "regular",
    })


def save(fig, name):
    IMG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  saved img/{name}")


# ---------------------------------------------------------------------------
# 1. Bar plots of renewal rate by genre + network (2 different visualizations)
#
# - Helps to visualize what the renewal rate looks like across different
#   networks and genres.
# - Shows the patterns between a genre and a network, which helps in the
#   overall renewability probability calculation.
# ---------------------------------------------------------------------------
def renewal_rate_bar(df, column, title, filename):
    # Each bar = share of shows in that group that got renewed
    rates = (df.groupby(column)["renewed"].mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.barplot(x=rates.index, y=rates.values, color=SAGE, ax=ax)
    ax.bar_label(ax.containers[0], fmt="%.0f%%", padding=3, color=CHARCOAL)
    ax.axhline(df["renewed"].mean() * 100, color=STONE, linestyle="--", linewidth=1)
    ax.text(len(rates) - 0.5, df["renewed"].mean() * 100 + 1.5, "all shows",
            ha="right", color=STONE, fontsize=9)
    ax.set(title=title, xlabel="", ylabel="Renewed", ylim=(0, 100))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="x", visible=False)
    save(fig, filename)


def conditional_probabilities_and_chi_square(df):
    """
    Conditional probabilities, like P(renewed | network = Netflix) vs
    P(renewed | network = CBS), and a chi-square test of whether the
    difference between networks is significant (or just chance).
    """
    p_by_network = df.groupby("network")["renewed"].mean()
    print("\nP(renewed | network):")
    print((p_by_network * 100).round(1).sort_values(ascending=False).to_string())

    # Chi-square test: is renewal independent of network?
    table = pd.crosstab(df["network"], df["renewed"])
    chi2, p_value, dof, _ = chi2_contingency(table)
    print(f"Chi-square (network vs renewed): chi2={chi2:.1f}, dof={dof}, p={p_value:.4g}")
    return p_by_network, p_value


# ---------------------------------------------------------------------------
# 2. Logistic regression model
#    The model learns from past shows what predicts renewal, and gives a
#    probability for each show.
# ---------------------------------------------------------------------------
def build_and_train(df, numeric=NUMERIC, verbose=True):
    df = df.copy()
    # Vote counts are very skewed, so use a log scale
    df["log_votes"] = np.log1p(df["votes"])

    # --- History ---------------------------------------------------------
    # Give the model the past data: rating, votes, network, genre,
    # S1 episode count, and whether the show was renewed / cancelled.
    X = df[numeric + CATEGORICAL]
    y = df["renewed"]

    # Hold some shows back. The model never sees them while learning, so the
    # ROC curve and calibration plot test it on shows it has not seen.
    # (Same random_state every time, so every version of the model is tested
    # on exactly the same shows.)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    # Numbers are put on the same scale; network and genre become 0/1 columns
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([("prep", prep), ("logreg", LogisticRegression(max_iter=1000))])

    # --- Weights ---------------------------------------------------------
    # The model adjusts the weights until its guesses match what happened:
    #   - Start with a guess (50% renewal for every show)
    #   - Check how wrong the guess is. Every show gets a renewal percentage,
    #     even the cancelled ones, so a high percentage on a cancelled show
    #     counts as a big mistake.
    #   - All the mistakes are added up: this is the loss
    #   - Nudge the weights to lower the loss. Example: if high-rated shows
    #     were renewed a lot more than predicted, the weight on rating goes up.
    #   - Repeat until improvement slows down
    # .fit() runs that whole loop for us.
    model.fit(X_train, y_train)

    if verbose:
        # Look at what it learned: positive weight = pushes toward renewal
        names = model.named_steps["prep"].get_feature_names_out()
        weights = pd.Series(model.named_steps["logreg"].coef_[0], index=names)
        print("\nLearned weights (positive = pushes toward renewal):")
        print(weights.sort_values(ascending=False).round(2).to_string())

    # --- Score a show ----------------------------------------------------
    # Score = sum of (clue x weight) for each clue.
    # High score = renewal, low score = cancellation.
    #
    # --- Score -> percentage ---------------------------------------------
    # The score is passed through the logistic function, which squeezes it
    # into a percentage between 0 and 1. predict_proba does both steps.
    # Column 1 = probability of "renewed".
    proba = model.predict_proba(X_test)[:, 1]

    # The Simulator and the "Pitch a new show" card use this percentage as
    # the renewal chance each season, then repeat the process to work out
    # how long a show will last (see the Monte Carlo step, coming next).
    return model, y_test, proba


# ---------------------------------------------------------------------------
# 3. ROC curve
#    After building the logistic regression model, shows how well the model
#    can tell renewed shows from cancelled ones: the accuracy of its
#    predictions. Closer to the top-left corner is better; the diagonal is
#    the same as guessing.
# ---------------------------------------------------------------------------
def roc_plot(y_test, proba, proba_no_votes=None):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)  # area under the curve: 0.5 = guessing, 1.0 = perfect

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(fpr, tpr, color=SAGE_DEEP, linewidth=2.2, label=f"Model (AUC {auc:.2f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color=SAGE)

    auc_nv = None
    if proba_no_votes is not None:
        # Same model, but with vote count removed, to see how much votes help
        fpr2, tpr2, _ = roc_curve(y_test, proba_no_votes)
        auc_nv = roc_auc_score(y_test, proba_no_votes)
        ax.plot(fpr2, tpr2, color="#b59a6d", linewidth=1.8, linestyle="-.",
                label=f"Without vote count (AUC {auc_nv:.2f})")

    ax.plot([0, 1], [0, 1], color=STONE, linestyle="--", linewidth=1.2, label="Random guessing")
    ax.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", frameon=False)
    save(fig, "roc.png")
    return auc, auc_nv


# ---------------------------------------------------------------------------
# 4. Calibration plot
#    Checks if the predicted chances are believable: compares the model's
#    percentage to the actual percentage.
#      - Get each show's prediction
#      - Group shows into buckets (0-10%, 10-20%, ... 90-100%)
#      - Calculate the real renewal rate: look at the shows in the bucket
#        and count how many were actually renewed (S1 to S2)
#      - Plot it
#    Points on the diagonal mean the percentages can be trusted.
# ---------------------------------------------------------------------------
def calibration_plot(y_test, proba, n_buckets=10, min_per_bucket=30):
    # Buckets hold the same NUMBER of shows each (lowest 10% of predictions,
    # next 10%, and so on). Equal-width buckets (0-10%, 10-20%...) can end up
    # with only a handful of shows, which makes the real rate jump around.
    n_buckets = max(3, min(n_buckets, len(y_test) // min_per_bucket))
    results = pd.DataFrame({"predicted": np.asarray(proba), "actual": np.asarray(y_test)})
    results["bucket"] = pd.qcut(results["predicted"], q=n_buckets, duplicates="drop")

    # For each bucket: the model's average percentage, the real share renewed,
    # and how many shows are in it
    groups = results.groupby("bucket", observed=True).agg(
        predicted=("predicted", "mean"), actual=("actual", "mean"), shows=("actual", "size"))
    print("\nCalibration buckets (predicted vs actual):")
    print(groups.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot([0, 1], [0, 1], color=STONE, linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax.plot(groups["predicted"], groups["actual"], color=SAGE_DEEP, linewidth=1.6, zorder=2)
    ax.scatter(groups["predicted"], groups["actual"], s=40 + 200 * groups["shows"] / groups["shows"].max(),
               color=SAGE_DEEP, zorder=3, label="Model (dot size = shows in bucket)")
    ax.set(title="Calibration plot", xlabel="Predicted renewal probability",
           ylabel="Actual share renewed", xlim=(0, 1), ylim=(0, 1.02))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "calibration.png")


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
def main():
    set_style()
    df, source = load_data(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Overall renewal rate: {df['renewed'].mean():.1%}")

    print("\nBar plots:")
    renewal_rate_bar(df, "genre", "Renewal rate by genre", "renewal_by_genre.png")
    renewal_rate_bar(df, "network", "Renewal rate by network", "renewal_by_network.png")
    p_by_network, chi_p = conditional_probabilities_and_chi_square(df)

    model, y_test, proba = build_and_train(df)

    # Same model again with vote count removed. Vote counts cover a show's whole
    # run, so longer-running shows have more of them. Comparing the two tells us
    # how much the model leans on votes.
    no_votes = [c for c in NUMERIC if c != "log_votes"]
    _, _, proba_no_votes = build_and_train(df, numeric=no_votes, verbose=False)

    print("\nROC and calibration:")
    auc, auc_no_votes = roc_plot(y_test, proba, proba_no_votes)
    calibration_plot(y_test, proba)
    print(f"\nAUC with votes: {auc:.3f}   AUC without votes: {auc_no_votes:.3f}")

    # Save the numbers so the website's stat tiles can use them later
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = {
        "data_source": source,
        "label_mode": LABEL_MODE if source != "synthetic demo data" else "n/a",
        "model_auc": round(float(auc), 3),
        "model_auc_without_votes": round(float(auc_no_votes), 3),
        "chi_square_p_value_network": float(chi_p),
        "p_renewed_by_network": {k: round(float(v), 3) for k, v in p_by_network.items()},
        "test_shows": int(len(y_test)),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Metrics saved to results/metrics.json")


if __name__ == "__main__":
    main()