import pandas as pd

from analytics.analysis import (
    completed_data,
    approval_mask,
    get_risk_thresholds,
)


def get_insights(df):

    data = completed_data(df)

    if data.empty:
        return []

    thresholds = get_risk_thresholds(df)

    insights = []

    # Overall approval rate
    approved = approval_mask(
        data["Loan_Approved"]
    )

    total_applications = len(data)

    approved_applications = int(
        approved.sum()
    )

    approval_rate = (
        approved_applications
        / total_applications
        * 100
        if total_applications
        else 0
    )

    insights.append({
        "title": "Overall Loan Approval",
        "fact": (
            f"{approved_applications} of "
            f"{total_applications} completed applications "
            f"were approved."
        ),
        "insight": (
            f"The overall approval rate is "
            f"{approval_rate:.1f}%."
        ),
        "action": (
            "Use this approval rate as the baseline "
            "for monitoring loan portfolio performance."
        ),
    })

    # Credit score insight
    credit_score = pd.to_numeric(
        data["Credit_Score"],
        errors="coerce"
    )

    credit_threshold = thresholds.get(
        "credit_score_low",
        thresholds.get(
            "low_credit",
            credit_score.quantile(0.25)
        )
    )

    low_credit_mask = (
        credit_score
        <= credit_threshold
    )

    higher_credit_mask = (
        credit_score
        > credit_threshold
    )

    low_credit_data = data.loc[
        low_credit_mask
    ]

    higher_credit_data = data.loc[
        higher_credit_mask
    ]

    low_credit_rate = 0

    higher_credit_rate = 0

    if len(low_credit_data) > 0:

        low_credit_rate = (
            approval_mask(
                low_credit_data[
                    "Loan_Approved"
                ]
            ).mean()
            * 100
        )

    if len(higher_credit_data) > 0:

        higher_credit_rate = (
            approval_mask(
                higher_credit_data[
                    "Loan_Approved"
                ]
            ).mean()
            * 100
        )

    insights.append({
        "title": "Credit Score is a Key Approval Driver",
        "fact": (
            f"Applicants at or below the lower credit "
            f"score threshold ({credit_threshold:.0f}) "
            f"have a {low_credit_rate:.1f}% approval rate, "
            f"compared with {higher_credit_rate:.1f}% for "
            f"applicants above the threshold."
        ),
        "insight": (
            "Credit score shows a strong association "
            "with loan approval in this dataset."
        ),
        "action": (
            "Use credit score as an important risk "
            "screening variable during applicant analysis."
        ),
    })

    # DTI insight
    dti = pd.to_numeric(
        data["DTI_Ratio"],
        errors="coerce"
    )

    dti_threshold = thresholds.get(
        "dti_high",
        thresholds.get(
            "high_dti",
            dti.quantile(0.75)
        )
    )

    high_dti_mask = (
        dti
        >= dti_threshold
    )

    lower_dti_mask = (
        dti
        < dti_threshold
    )

    high_dti_data = data.loc[
        high_dti_mask
    ]

    lower_dti_data = data.loc[
        lower_dti_mask
    ]

    high_dti_rate = 0

    lower_dti_rate = 0

    if len(high_dti_data) > 0:

        high_dti_rate = (
            approval_mask(
                high_dti_data[
                    "Loan_Approved"
                ]
            ).mean()
            * 100
        )

    if len(lower_dti_data) > 0:

        lower_dti_rate = (
            approval_mask(
                lower_dti_data[
                    "Loan_Approved"
                ]
            ).mean()
            * 100
        )

    insights.append({
        "title": "High DTI Indicates Higher Risk",
        "fact": (
            f"Applicants at or above the DTI threshold "
            f"({dti_threshold:.2f}) have a "
            f"{high_dti_rate:.1f}% approval rate, "
            f"compared with {lower_dti_rate:.1f}% "
            f"for applicants below the threshold."
        ),
        "insight": (
            "Higher debt-to-income levels are strongly "
            "associated with lower loan approval in "
            "this dataset."
        ),
        "action": (
            "Monitor DTI carefully when assessing "
            "applicant repayment risk."
        ),
    })

    # Combined risk insight
    low_credit = (
        credit_score
        <= credit_threshold
    )

    high_dti = (
        dti
        >= dti_threshold
    )

    existing_loans = pd.to_numeric(
        data["Existing_Loans"],
        errors="coerce"
    )

    existing_loans_threshold = thresholds.get(
        "existing_loans_high",
        thresholds.get(
            "high_existing_loans",
            existing_loans.median()
        )
    )

    high_existing_loans = (
        existing_loans
        > existing_loans_threshold
    )

    risk_factor_count = (
        low_credit.astype(int)
        + high_dti.astype(int)
        + high_existing_loans.astype(int)
    )

    high_risk_mask = (
        risk_factor_count >= 2
    )

    high_risk_data = data.loc[
        high_risk_mask
    ]

    high_risk_applications = len(
        high_risk_data
    )

    high_risk_approval_rate = 0

    if high_risk_applications > 0:

        high_risk_approval_rate = (
            approval_mask(
                high_risk_data[
                    "Loan_Approved"
                ]
            ).mean()
            * 100
        )

    insights.append({
        "title": "Multiple Risk Factors Increase Exposure",
        "fact": (
            f"{high_risk_applications} applications "
            f"have at least two identified risk factors. "
            f"Their approval rate is "
            f"{high_risk_approval_rate:.1f}%."
        ),
        "insight": (
            "Applications with multiple risk indicators "
            "show substantially lower approval levels "
            "in this dataset."
        ),
        "action": (
            "Review combinations of credit score, DTI, "
            "and existing loans instead of relying on "
            "one metric alone."
        ),
    })

    return insights