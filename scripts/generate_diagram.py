"""Generate Loan-Lens system architecture diagram."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs("screenshots", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#0f172a")

COLORS = {
    "api": "#3b82f6",
    "ml": "#10b981",
    "db": "#f59e0b",
    "monitor": "#8b5cf6",
    "pipeline": "#ef4444",
    "client": "#6b7280",
    "arrow": "#94a3b8",
}


def box(ax, x, y, w, h, label, sublabel, color):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.65, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    ax.text(x + w / 2, y + h * 0.3, sublabel, ha="center", va="center",
            fontsize=7, color="white", alpha=0.85)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.5))


# Title
ax.text(8, 8.5, "Loan-Lens — System Architecture", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")

# Client
box(ax, 0.3, 5.5, 2.2, 1.2, "Client", "REST HTTP", COLORS["client"])

# FastAPI
box(ax, 3.2, 5.5, 2.8, 1.8, "FastAPI API", "/predict /health\n/metrics /drift", COLORS["api"])

# ML Model
box(ax, 7.0, 6.5, 2.8, 1.5, "Ensemble Model", "XGBoost+LightGBM+RF\n5-fold CV · AUC", COLORS["ml"])

# Feature Engineering
box(ax, 7.0, 4.5, 2.8, 1.5, "Feature Pipeline", "26 features · sklearn\nRatios · Bins · Logs", COLORS["ml"])

# PostgreSQL
box(ax, 3.2, 2.8, 2.8, 1.5, "PostgreSQL", "prediction_logs\ndrift_logs", COLORS["db"])

# Monitoring
box(ax, 7.0, 2.8, 2.8, 1.5, "Monitoring", "KS-test Drift\nPrediction Stats", COLORS["monitor"])

# Retraining
box(ax, 11.0, 4.5, 3.0, 1.5, "Retrain Pipeline", "Airflow DAG\nweekly schedule", COLORS["pipeline"])

# MLflow
box(ax, 11.0, 2.5, 3.0, 1.5, "Model Registry", "metrics.json\nmodel.joblib", COLORS["ml"])

# Arrows
arrow(ax, 2.5, 6.1, 3.2, 6.4)
arrow(ax, 6.0, 6.4, 7.0, 7.0)
arrow(ax, 6.0, 6.0, 7.0, 5.2)
arrow(ax, 6.0, 5.8, 3.2, 3.8)
arrow(ax, 6.0, 3.5, 7.0, 3.5)
arrow(ax, 9.8, 6.5, 11.0, 5.5)
arrow(ax, 9.8, 3.5, 11.0, 3.2)
arrow(ax, 14.0, 4.5, 14.0, 4.0)

plt.tight_layout()
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Architecture diagram saved to screenshots/architecture.png")
