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
Put your real data in data/tv_shows.csv with these columns:
    title, network, genre, imdb_rating, votes, s1_episodes, renewed
where `renewed` is 1 if the show got a season 2 and 0 if it was cancelled
after season 1 (the S1 to S2 decision).

If that file does not exist, the script builds a SYNTHETIC demo dataset so
the whole pipeline still runs. Charts made from demo data are for layout
only and do not say anything real about TV. Replace the data before you
publish any of the numbers.
"""

import json
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

DATA_PATH = Path("data/tv_shows.csv")
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


def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        needed = {"network", "genre", "imdb_rating", "votes", "s1_episodes", "renewed"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{DATA_PATH} is missing columns: {sorted(missing)}")
        df = df.dropna(subset=list(needed))
        print(f"Loaded {len(df)} shows from {DATA_PATH}")
        return df
    print("WARNING: data/tv_shows.csv not found. Using SYNTHETIC demo data.")
    return make_demo_data()


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
def build_and_train(df):
    df = df.copy()
    # Vote counts are very skewed, so use a log scale
    df["log_votes"] = np.log1p(df["votes"])

    # --- History ---------------------------------------------------------
    # Give the model the past data: rating, votes, network, genre,
    # S1 episode count, and whether the show was renewed / cancelled.
    X = df[NUMERIC + CATEGORICAL]
    y = df["renewed"]

    # Hold some shows back. The model never sees them while learning, so the
    # ROC curve and calibration plot test it on shows it has not seen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    # Numbers are put on the same scale; network and genre become 0/1 columns
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
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
def roc_plot(y_test, proba):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)  # area under the curve: 0.5 = guessing, 1.0 = perfect

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(fpr, tpr, color=SAGE_DEEP, linewidth=2.2, label=f"Model (AUC {auc:.2f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color=SAGE)
    ax.plot([0, 1], [0, 1], color=STONE, linestyle="--", linewidth=1.2, label="Random guessing")
    ax.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", frameon=False)
    save(fig, "roc.png")
    return auc


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
def calibration_plot(y_test, proba):
    # actual_rate = real share renewed in each bucket
    # mean_predicted = the model's average percentage in that bucket
    actual_rate, mean_predicted = calibration_curve(
        y_test, proba, n_bins=10, strategy="uniform")  # empty buckets are skipped

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot([0, 1], [0, 1], color=STONE, linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax.plot(mean_predicted, actual_rate, marker="o", color=SAGE_DEEP, linewidth=2,
            markersize=7, label="Model")
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
    df = load_data()
    print(f"Overall renewal rate: {df['renewed'].mean():.1%}")

    print("\nBar plots:")
    renewal_rate_bar(df, "genre", "Renewal rate by genre", "renewal_by_genre.png")
    renewal_rate_bar(df, "network", "Renewal rate by network", "renewal_by_network.png")
    p_by_network, chi_p = conditional_probabilities_and_chi_square(df)

    model, y_test, proba = build_and_train(df)

    print("\nROC and calibration:")
    auc = roc_plot(y_test, proba)
    calibration_plot(y_test, proba)

    # Save the numbers so the website's stat tiles can use them later
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = {
        "model_auc": round(float(auc), 3),
        "chi_square_p_value_network": float(chi_p),
        "p_renewed_by_network": {k: round(float(v), 3) for k, v in p_by_network.items()},
        "test_shows": int(len(y_test)),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nModel AUC: {auc:.2f}. Metrics saved to results/metrics.json")


if __name__ == "__main__":
    main()