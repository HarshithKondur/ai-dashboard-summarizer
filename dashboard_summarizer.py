"""
AI Dashboard Summarizer
Loads Car_Insurance_Claim.csv, generates 4 visualizations,
extracts key statistics, and sends them to Claude API for
an executive summary.
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import anthropic

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CSV_PATH = "Car_Insurance_Claim.csv"
OUTPUT_DIR = "charts"
MODEL = "claude-sonnet-4-6"

CHART_PATHS = {
    "vehicle_type": os.path.join(OUTPUT_DIR, "claims_by_vehicle_type.png"),
    "age_group":    os.path.join(OUTPUT_DIR, "claim_rate_by_age_group.png"),
    "gender":       os.path.join(OUTPUT_DIR, "claims_by_gender.png"),
    "outcome":      os.path.join(OUTPUT_DIR, "outcome_breakdown.png"),
}

# Colour palette
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[✓] Loaded {len(df):,} rows from '{path}'")
    return df


def save_chart(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────
# Chart 1 – Claims count by vehicle type
# ─────────────────────────────────────────────
def chart_vehicle_type(df: pd.DataFrame) -> dict:
    """Bar chart: number of claims (OUTCOME == 1) per VEHICLE_TYPE."""
    claims = df[df["OUTCOME"] == 1]
    counts = claims["VEHICLE_TYPE"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE[:len(counts)], edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=11)
    ax.set_title("Claims Count by Vehicle Type", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Vehicle Type", fontsize=11)
    ax.set_ylabel("Number of Claims", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    save_chart(fig, CHART_PATHS["vehicle_type"])

    stats = {
        "total_claims": int(counts.sum()),
        "by_vehicle_type": counts.to_dict(),
        "highest_vehicle": counts.idxmax(),
        "highest_count":   int(counts.max()),
        "lowest_vehicle":  counts.idxmin(),
        "lowest_count":    int(counts.min()),
    }
    return stats


# ─────────────────────────────────────────────
# Chart 2 – Claim rate by age group
# ─────────────────────────────────────────────
def chart_age_group(df: pd.DataFrame) -> dict:
    """Horizontal bar chart: claim rate (%) per AGE group."""
    age_order = ["16-25", "26-39", "40-64", "65+"]
    age_stats = (
        df.groupby("AGE")["OUTCOME"]
        .agg(total="count", claims="sum")
        .reindex(age_order)
    )
    age_stats["claim_rate_pct"] = (age_stats["claims"] / age_stats["total"] * 100).round(2)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        age_stats.index, age_stats["claim_rate_pct"],
        color=PALETTE[:len(age_stats)], edgecolor="white"
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11)
    ax.set_title("Claim Rate (%) by Age Group", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Claim Rate (%)", fontsize=11)
    ax.set_ylabel("Age Group", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, age_stats["claim_rate_pct"].max() * 1.15)
    plt.tight_layout()
    save_chart(fig, CHART_PATHS["age_group"])

    stats = {
        "age_claim_rates_pct": age_stats["claim_rate_pct"].to_dict(),
        "age_totals":          age_stats["total"].to_dict(),
        "age_claims":          age_stats["claims"].to_dict(),
        "highest_rate_age":    age_stats["claim_rate_pct"].idxmax(),
        "highest_rate_pct":    float(age_stats["claim_rate_pct"].max()),
        "lowest_rate_age":     age_stats["claim_rate_pct"].idxmin(),
        "lowest_rate_pct":     float(age_stats["claim_rate_pct"].min()),
    }
    return stats


# ─────────────────────────────────────────────
# Chart 3 – Claims by gender
# ─────────────────────────────────────────────
def chart_gender(df: pd.DataFrame) -> dict:
    """Grouped bar chart: claims vs no-claims per GENDER."""
    gender_stats = df.groupby(["GENDER", "OUTCOME"]).size().unstack(fill_value=0)
    gender_stats.columns = ["No Claim", "Claim"]

    x = range(len(gender_stats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_no  = ax.bar([i - width/2 for i in x], gender_stats["No Claim"], width, label="No Claim", color=PALETTE[0])
    bars_yes = ax.bar([i + width/2 for i in x], gender_stats["Claim"],    width, label="Claim",    color=PALETTE[1])
    ax.bar_label(bars_no,  fmt="%d", padding=4, fontsize=10)
    ax.bar_label(bars_yes, fmt="%d", padding=4, fontsize=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels(gender_stats.index, fontsize=11)
    ax.set_title("Claims vs No-Claims by Gender", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Gender", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_chart(fig, CHART_PATHS["gender"])

    total_by_gender = gender_stats.sum(axis=1)
    claim_rate_by_gender = (gender_stats["Claim"] / total_by_gender * 100).round(2)

    stats = {
        "gender_claims":         gender_stats["Claim"].to_dict(),
        "gender_no_claims":      gender_stats["No Claim"].to_dict(),
        "gender_claim_rate_pct": claim_rate_by_gender.to_dict(),
        "higher_claim_gender":   claim_rate_by_gender.idxmax(),
        "higher_claim_rate_pct": float(claim_rate_by_gender.max()),
    }
    return stats


# ─────────────────────────────────────────────
# Chart 4 – Fraud vs Legitimate Claims Breakdown
# ─────────────────────────────────────────────
def chart_outcome(df: pd.DataFrame) -> dict:
    """Pie chart: overall OUTCOME distribution (0 = No Claim, 1 = Claim)."""
    outcome_counts = df["OUTCOME"].value_counts().sort_index()
    labels = {0: "No Claim (Legitimate)", 1: "Claim Filed"}
    label_text = [labels.get(k, str(k)) for k in outcome_counts.index]
    total = outcome_counts.sum()
    pct   = (outcome_counts / total * 100).round(2)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        outcome_counts.values,
        labels=label_text,
        autopct="%1.1f%%",
        startangle=140,
        colors=[PALETTE[0], PALETTE[2]],
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
    ax.set_title("Claims vs No-Claims Breakdown", fontsize=14, fontweight="bold", pad=16)
    plt.tight_layout()
    save_chart(fig, CHART_PATHS["outcome"])

    stats = {
        "total_records":     int(total),
        "no_claim_count":    int(outcome_counts.get(0, 0)),
        "claim_count":       int(outcome_counts.get(1, 0)),
        "no_claim_pct":      float(pct.get(0, 0)),
        "claim_pct":         float(pct.get(1, 0)),
        "claim_to_no_claim_ratio": round(
            outcome_counts.get(1, 0) / max(outcome_counts.get(0, 1), 1), 4
        ),
    }
    return stats


# ─────────────────────────────────────────────
# Claude API – Executive Summary
# ─────────────────────────────────────────────
def build_prompt(stats: dict) -> str:
    vt = stats["vehicle_type"]
    ag = stats["age_group"]
    gd = stats["gender"]
    oc = stats["outcome"]

    prompt = f"""
You are a senior insurance analytics consultant. Below are key statistics extracted from a
car insurance claims dataset containing {oc['total_records']:,} customer records.
Please write a clear, concise executive summary (4–6 paragraphs) for a non-technical
audience — a C-suite leadership team — explaining what the data reveals about claims
behaviour, risk patterns, and any actionable business insights.

=== CHART 1: Claims Count by Vehicle Type ===
Total claims filed: {vt['total_claims']:,}
Breakdown by vehicle type: {vt['by_vehicle_type']}
Highest claims: {vt['highest_vehicle']} ({vt['highest_count']:,} claims)
Lowest claims:  {vt['lowest_vehicle']}  ({vt['lowest_count']:,} claims)

=== CHART 2: Claim Rate (%) by Age Group ===
Claim rates by age group: {ag['age_claim_rates_pct']}
Number of customers per age group: {ag['age_totals']}
Highest claim-rate age group: {ag['highest_rate_age']} ({ag['highest_rate_pct']}%)
Lowest claim-rate age group:  {ag['lowest_rate_age']}  ({ag['lowest_rate_pct']}%)

=== CHART 3: Claims by Gender ===
Number of claims by gender: {gd['gender_claims']}
Number of no-claims by gender: {gd['gender_no_claims']}
Claim rates by gender: {gd['gender_claim_rate_pct']}
Gender with higher claim rate: {gd['higher_claim_gender']} ({gd['higher_claim_rate_pct']}%)

=== CHART 4: Overall Claims vs No-Claims Breakdown ===
Total records: {oc['total_records']:,}
No-claim customers: {oc['no_claim_count']:,} ({oc['no_claim_pct']}%)
Claim customers:    {oc['claim_count']:,} ({oc['claim_pct']}%)
Claim-to-no-claim ratio: {oc['claim_to_no_claim_ratio']}

Please focus on:
1. Key risk segments (age, gender, vehicle type)
2. Overall portfolio health (claims vs no-claims)
3. Actionable recommendations for underwriting and pricing
""".strip()
    return prompt


def get_executive_summary(stats: dict) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[!] ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("    Set it with:  export ANTHROPIC_API_KEY='your-api-key-here'")
        print("    Then re-run:  python3 dashboard_summarizer.py\n")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(stats)

    print(f"\n[→] Sending statistics to Claude ({MODEL}) for executive summary …\n")
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AI Dashboard Summarizer – Car Insurance Claims")
    print("=" * 60)

    # 1. Load data
    df = load_data(CSV_PATH)

    # 2. Generate charts & collect stats
    print("\n[→] Generating visualizations …")
    stats = {}

    print("  [1/4] Claims by vehicle type …")
    stats["vehicle_type"] = chart_vehicle_type(df)

    print("  [2/4] Claim rate by age group …")
    stats["age_group"] = chart_age_group(df)

    print("  [3/4] Claims by gender …")
    stats["gender"] = chart_gender(df)

    print("  [4/4] Outcome breakdown (claims vs no-claims) …")
    stats["outcome"] = chart_outcome(df)

    print(f"\n[✓] All 4 charts saved to '{OUTPUT_DIR}/'")

    # 3. Get AI executive summary
    summary = get_executive_summary(stats)

    # 4. Print summary
    print("\n" + "=" * 60)
    print("  EXECUTIVE SUMMARY  (generated by Claude)")
    print("=" * 60)
    print(summary)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
