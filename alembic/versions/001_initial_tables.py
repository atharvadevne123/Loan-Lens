"""Initial tables: prediction_logs and drift_logs.

Revision ID: 001
Revises:
Create Date: 2025-05-13
"""

import sqlalchemy as sa

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("correlation_id", sa.String, nullable=False, index=True),
        sa.Column("input_features", sa.JSON, nullable=False),
        sa.Column("probability", sa.Float, nullable=False),
        sa.Column("prediction", sa.Integer, nullable=False),
        sa.Column("model_version", sa.String, default="1.0.0"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("feature", sa.String, nullable=False),
        sa.Column("ks_statistic", sa.Float, nullable=False),
        sa.Column("p_value", sa.Float, nullable=False),
        sa.Column("drift_detected", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("drift_logs")
    op.drop_table("prediction_logs")
