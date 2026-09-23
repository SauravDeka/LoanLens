from pathlib import Path

import pandas as pd
import plotly.express as px


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_approval_data.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.strip()

    return df


def completed_data(df):
    data = df.copy()

    if "Loan_Approved" not in data.columns:
        raise ValueError(
            "Loan_Approved column is missing from the dataset."
        )

    data = data.dropna(
        subset=["Loan_Approved"]
    ).copy()

    return data


def approval_mask(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "1", "approved", "true"])
    )


def get_kpis(df):
    data = df.copy()

    total_applications = len(data)

    completed = completed_data(data)

    approved_mask = approval_mask(
        completed["Loan_Approved"]
    )

    approved_applications = int(
        approved_mask.sum()
    )

    rejected_applications = int(
        len(completed) - approved_applications
    )

    completed_applications = len(completed)

    if completed_applications > 0:
        approval_rate = (
            approved_applications
            / completed_applications
            * 100
        )
    else:
        approval_rate = 0

    return {
        "total_applications": total_applications,
        "completed_applications": completed_applications,
        "approved_applications": approved_applications,
        "rejected_applications": rejected_applications,
        "approval_rate": round(
            approval_rate,
            1
        ),
        "average_loan_amount": round(
            pd.to_numeric(
                data["Loan_Amount"],
                errors="coerce"
            ).mean(),
            2
        ),
        "average_credit_score": round(
            pd.to_numeric(
                data["Credit_Score"],
                errors="coerce"
            ).mean(),
            1
        ),
        "average_dti": round(
            pd.to_numeric(
                data["DTI_Ratio"],
                errors="coerce"
            ).mean(),
            2
        ),
        "average_income": round(
            pd.to_numeric(
                data["Applicant_Income"],
                errors="coerce"
            ).mean(),
            2
        ),
    }


def approval_by_category(df, column):
    data = completed_data(df)

    if column not in data.columns:
        return pd.DataFrame(
            columns=[
                column,
                "Applications",
                "Approved",
                "Approval Rate",
            ]
        )

    result = []

    for category, group in data.groupby(
        column,
        dropna=False
    ):

        approved = approval_mask(
            group["Loan_Approved"]
        ).sum()

        applications = len(group)

        rate = (
            approved / applications * 100
            if applications
            else 0
        )

        result.append(
            {
                column: (
                    "Unknown"
                    if pd.isna(category)
                    else str(category)
                ),
                "Applications": applications,
                "Approved": int(approved),
                "Approval Rate": round(
                    rate,
                    1
                ),
            }
        )

    return pd.DataFrame(result)


def create_quartile_analysis(df, column):
    data = completed_data(df)

    if column not in data.columns:
        return pd.DataFrame()

    values = pd.to_numeric(
        data[column],
        errors="coerce"
    )

    working = data.copy()
    working["_value"] = values

    working = working.dropna(
        subset=["_value"]
    )

    if len(working) < 4:
        return pd.DataFrame()

    try:
        working["Quartile"] = pd.qcut(
            working["_value"],
            q=4,
            labels=[
                "Q1",
                "Q2",
                "Q3",
                "Q4",
            ],
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()

    rows = []

    for quartile, group in working.groupby(
        "Quartile",
        observed=False
    ):

        approved = approval_mask(
            group["Loan_Approved"]
        ).sum()

        applications = len(group)

        rate = (
            approved / applications * 100
            if applications
            else 0
        )

        rows.append(
            {
                "Quartile": str(quartile),
                "Applications": applications,
                "Approved": int(approved),
                "Approval Rate": round(
                    rate,
                    1
                ),
                "Average Value": round(
                    group["_value"].mean(),
                    2
                ),
            }
        )

    return pd.DataFrame(rows)


def get_risk_thresholds(df):
    data = completed_data(df)

    credit_score = pd.to_numeric(
        data["Credit_Score"],
        errors="coerce"
    )

    dti = pd.to_numeric(
        data["DTI_Ratio"],
        errors="coerce"
    )

    existing_loans = pd.to_numeric(
        data["Existing_Loans"],
        errors="coerce"
    )

    return {
        "credit_score_low": float(
            credit_score.quantile(0.25)
        ),
        "dti_high": float(
            dti.quantile(0.75)
        ),
        "existing_loans_high": float(
            existing_loans.median()
        ),
    }


def get_risk_summary(df):
    data = completed_data(df)

    thresholds = get_risk_thresholds(df)

    credit_score = pd.to_numeric(
        data["Credit_Score"],
        errors="coerce"
    )

    dti = pd.to_numeric(
        data["DTI_Ratio"],
        errors="coerce"
    )

    existing_loans = pd.to_numeric(
        data["Existing_Loans"],
        errors="coerce"
    )

    low_credit = (
        credit_score
        <= thresholds["credit_score_low"]
    )

    high_dti = (
        dti
        >= thresholds["dti_high"]
    )

    high_existing_loans = (
        existing_loans
        > thresholds["existing_loans_high"]
    )

    return {
        "low_credit": {
            "count": int(low_credit.sum()),
            "approval_rate": round(
                approval_mask(
                    data.loc[
                        low_credit,
                        "Loan_Approved"
                    ]
                ).mean()
                * 100,
                1
            )
            if low_credit.sum()
            else 0,
        },

        "high_dti": {
            "count": int(high_dti.sum()),
            "approval_rate": round(
                approval_mask(
                    data.loc[
                        high_dti,
                        "Loan_Approved"
                    ]
                ).mean()
                * 100,
                1
            )
            if high_dti.sum()
            else 0,
        },

        "high_existing_loans": {
            "count": int(
                high_existing_loans.sum()
            ),
            "approval_rate": round(
                approval_mask(
                    data.loc[
                        high_existing_loans,
                        "Loan_Approved"
                    ]
                ).mean()
                * 100,
                1
            )
            if high_existing_loans.sum()
            else 0,
        },
    }


def get_combined_risk_analysis(df):
    data = completed_data(df)

    thresholds = get_risk_thresholds(df)

    credit_score = pd.to_numeric(
        data["Credit_Score"],
        errors="coerce"
    )

    dti = pd.to_numeric(
        data["DTI_Ratio"],
        errors="coerce"
    )

    existing_loans = pd.to_numeric(
        data["Existing_Loans"],
        errors="coerce"
    )

    low_credit = (
        credit_score
        <= thresholds["credit_score_low"]
    )

    high_dti = (
        dti
        >= thresholds["dti_high"]
    )

    high_existing_loans = (
        existing_loans
        > thresholds["existing_loans_high"]
    )

    risk_count = (
        low_credit.astype(int)
        + high_dti.astype(int)
        + high_existing_loans.astype(int)
    )

    rows = []

    for number in range(4):

        mask = risk_count == number

        applications = int(mask.sum())

        if applications:
            approved = int(
                approval_mask(
                    data.loc[
                        mask,
                        "Loan_Approved"
                    ]
                ).sum()
            )

            rate = (
                approved
                / applications
                * 100
            )
        else:
            approved = 0
            rate = 0

        rows.append(
            {
                "Risk Factors": number,
                "Applications": applications,
                "Approved": approved,
                "Approval Rate": round(
                    rate,
                    1
                ),
            }
        )

    return pd.DataFrame(rows)


def apply_chart_style(fig, title):
    fig.update_layout(
        title={
            "text": title,
            "x": 0,
            "xanchor": "left",
            "font": {
                "size": 16
            },
        },
        height=300,
        margin={
            "l": 40,
            "r": 20,
            "t": 55,
            "b": 40,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={
            "family": "Arial",
            "size": 12,
        },
        showlegend=False,
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        gridcolor="#eeeeee"
    )

    return fig


def chart_html(fig):
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": False
        }
    )


def create_approval_band_chart(df):
    data = create_quartile_analysis(
        df,
        "Credit_Score"
    )

    if data.empty:
        return ""

    fig = px.bar(
        data,
        x="Quartile",
        y="Approval Rate",
        text="Approval Rate",
        labels={
            "Approval Rate": "Approval Rate (%)"
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    return chart_html(
        apply_chart_style(
            fig,
            "Approval Rate by Credit Score Band"
        )
    )


def create_credit_score_chart(df):
    data = create_quartile_analysis(
        df,
        "Credit_Score"
    )

    if data.empty:
        return ""

    fig = px.bar(
        data,
        x="Quartile",
        y="Approval Rate",
        text="Approval Rate",
        labels={
            "Approval Rate": "Approval Rate (%)"
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    return chart_html(
        apply_chart_style(
            fig,
            "Approval Rate by Credit Score"
        )
    )


def create_dti_chart(df):
    data = create_quartile_analysis(
        df,
        "DTI_Ratio"
    )

    if data.empty:
        return ""

    fig = px.bar(
        data,
        x="Quartile",
        y="Approval Rate",
        text="Approval Rate",
        labels={
            "Approval Rate": "Approval Rate (%)"
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    return chart_html(
        apply_chart_style(
            fig,
            "Approval Rate by DTI Band"
        )
    )


def create_income_chart(df):
    data = create_quartile_analysis(
        df,
        "Applicant_Income"
    )

    if data.empty:
        return ""

    fig = px.bar(
        data,
        x="Quartile",
        y="Approval Rate",
        text="Approval Rate",
        labels={
            "Approval Rate": "Approval Rate (%)"
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    return chart_html(
        apply_chart_style(
            fig,
            "Approval Rate by Applicant Income"
        )
    )


def create_loan_amount_chart(df):
    data = create_quartile_analysis(
        df,
        "Loan_Amount"
    )

    if data.empty:
        return ""

    fig = px.bar(
        data,
        x="Quartile",
        y="Approval Rate",
        text="Approval Rate",
        labels={
            "Approval Rate": "Approval Rate (%)"
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    return chart_html(
        apply_chart_style(
            fig,
            "Approval Rate by Loan Amount"
        )
    )


def create_employment_chart(df):
    data = approval_by_category(
        df,
        "Employment_Status"
    )

    if data.empty:
        return ""

    fig = px.bar(
        data,
        x="Employment_Status",
        y="Approval Rate",
        text="Approval Rate",
        labels={
            "Approval Rate": "Approval Rate (%)"
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    return chart_html(
        apply_chart_style(
            fig,
            "Approval Rate by Employment Status"
        )
    )


def create_loan_purpose_chart(df):
    data = approval_by_category(
        df,
        "Loan_Purpose"
    )

    if data.empty:
        return ""

    fig = px.bar(
        data,
        x="Loan_Purpose",
        y="Approval Rate",
        text="Approval Rate",
        labels={
            "Approval Rate": "Approval Rate (%)"
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    return chart_html(
        apply_chart_style(
            fig,
            "Approval Rate by Loan Purpose"
        )
    )


def create_risk_factor_chart(df):
    data = get_combined_risk_analysis(
        df
    )

    if data.empty:
        return ""

    fig = px.bar(
        data,
        x="Risk Factors",
        y="Approval Rate",
        text="Approval Rate",
        labels={
            "Approval Rate": "Approval Rate (%)"
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    return chart_html(
        apply_chart_style(
            fig,
            "Approval Rate by Number of Risk Factors"
        )
    )


def create_combined_risk_chart(df):
    data = get_combined_risk_analysis(
        df
    )

    if data.empty:
        return ""

    fig = px.bar(
        data,
        x="Risk Factors",
        y="Applications",
        text="Applications",
        labels={
            "Applications": "Applications"
        },
    )

    fig.update_traces(
        textposition="outside"
    )

    return chart_html(
        apply_chart_style(
            fig,
            "Applications by Risk Factor Count"
        )
    )