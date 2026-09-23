from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    session,
)
import traceback
import pandas as pd

from analytics.analysis import (
    load_data,
    get_kpis,
    get_risk_summary,
    get_combined_risk_analysis,
    create_credit_score_chart,
    create_dti_chart,
    create_income_chart,
    create_loan_amount_chart,
    create_employment_chart,
    create_loan_purpose_chart,
    create_risk_factor_chart,
    create_combined_risk_chart,
)

from analytics.model import evaluate_random_forest
from src.credit_wise.predict import predict_loan


app = Flask(__name__)
app.secret_key = "loanlens-secret-key"


def make_category_analysis(df, column):
    data = df.dropna(
        subset=["Loan_Approved"]
    ).copy()

    rows = []

    for category, group in data.groupby(column):
        applications = len(group)

        approved = (
            group["Loan_Approved"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
            .sum()
        )

        approval_rate = (
            approved / applications * 100
            if applications > 0
            else 0
        )

        rows.append(
            {
                column: str(category),
                "Applications": int(applications),
                "Approved": int(approved),
                "Approval_Rate": round(
                    float(approval_rate),
                    1,
                ),
            }
        )

    return rows


def create_project_insights(df):
    data = df.dropna(
        subset=["Loan_Approved"]
    ).copy()

    approved = (
        data["Loan_Approved"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
    )

    total = len(data)
    approved_count = int(approved.sum())

    approval_rate = (
        approved_count / total * 100
        if total > 0
        else 0
    )

    credit_threshold = float(
        data["Credit_Score"].median()
    )

    low_credit = data[
        data["Credit_Score"] <= credit_threshold
    ]

    high_credit = data[
        data["Credit_Score"] > credit_threshold
    ]

    low_credit_rate = (
        low_credit["Loan_Approved"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .mean()
        * 100
        if len(low_credit) > 0
        else 0
    )

    high_credit_rate = (
        high_credit["Loan_Approved"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .mean()
        * 100
        if len(high_credit) > 0
        else 0
    )

    dti_threshold = float(
        data["DTI_Ratio"].median()
    )

    low_dti = data[
        data["DTI_Ratio"] <= dti_threshold
    ]

    high_dti = data[
        data["DTI_Ratio"] > dti_threshold
    ]

    low_dti_rate = (
        low_dti["Loan_Approved"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .mean()
        * 100
        if len(low_dti) > 0
        else 0
    )

    high_dti_rate = (
        high_dti["Loan_Approved"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .mean()
        * 100
        if len(high_dti) > 0
        else 0
    )

    return [
        {
            "title": "Overall Loan Approval",
            "fact": (
                f"{approved_count} of {total} completed "
                "applications were approved."
            ),
            "insight": (
                f"The overall approval rate is "
                f"{approval_rate:.1f}%."
            ),
            "action": (
                "Use this approval rate as the baseline "
                "for monitoring loan portfolio performance."
            ),
        },
        {
            "title": "Credit Score is a Key Approval Driver",
            "fact": (
                f"Applicants at or below the median credit "
                f"score ({credit_threshold:.0f}) have a "
                f"{low_credit_rate:.1f}% approval rate, "
                f"compared with {high_credit_rate:.1f}% "
                "above the median."
            ),
            "insight": (
                "Credit score shows a strong association "
                "with loan approval in this dataset."
            ),
            "action": (
                "Use credit score as an important risk "
                "screening variable during applicant analysis."
            ),
        },
        {
            "title": "Higher DTI Indicates Higher Risk",
            "fact": (
                f"Applicants above the median DTI "
                f"({dti_threshold:.2f}) have a "
                f"{high_dti_rate:.1f}% approval rate, "
                f"compared with {low_dti_rate:.1f}% "
                "below the median."
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
        },
    ]


def normalize_applicant(applicant):
    data = dict(applicant)

    integer_fields = [
        "Age",
        "Dependents",
        "Existing_Loans",
        "Loan_Term",
    ]

    float_fields = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Savings",
        "Credit_Score",
        "DTI_Ratio",
        "Loan_Amount",
        "Collateral_Value",
    ]

    for field in integer_fields:
        if field not in data:
            raise ValueError(
                f"Missing required field: {field}"
            )

        value = str(data[field]).strip()

        if value == "":
            raise ValueError(
                f"Missing required field: {field}"
            )

        data[field] = int(float(value))

    for field in float_fields:
        if field not in data:
            raise ValueError(
                f"Missing required field: {field}"
            )

        value = str(data[field]).strip()

        if value == "":
            raise ValueError(
                f"Missing required field: {field}"
            )

        data[field] = float(value)

    text_fields = [
        "Gender",
        "Marital_Status",
        "Education_Level",
        "Employment_Status",
        "Employer_Category",
        "Loan_Purpose",
        "Property_Area",
    ]

    for field in text_fields:
        if field not in data:
            raise ValueError(
                f"Missing required field: {field}"
            )

        data[field] = str(data[field]).strip()

    data["Applicant_ID"] = str(
        data.get(
            "Applicant_ID",
            "WEB001",
        )
    )

    return data


def template_value(value):
    """
    Convert Pandas objects into normal Python objects
    before sending them to Jinja.
    """
    if isinstance(value, pd.DataFrame):
        return value.to_dict(
            orient="records"
        )

    if isinstance(value, pd.Series):
        return value.to_dict()

    if value is None:
        return []

    return value


@app.route("/")
def home():
    return redirect(
        url_for("dashboard")
    )


@app.route("/dashboard")
def dashboard():
    df = load_data()

    kpis = get_kpis(df)

    credit_score_chart = (
        create_credit_score_chart(df)
    )

    dti_chart = create_dti_chart(df)

    income_chart = create_income_chart(df)

    loan_amount_chart = (
        create_loan_amount_chart(df)
    )

    employment_chart = (
        create_employment_chart(df)
    )

    loan_purpose_chart = (
        create_loan_purpose_chart(df)
    )

    risk_summary = get_risk_summary(df)

    combined_risk_analysis = (
        get_combined_risk_analysis(df)
    )

    risk_factor_chart = (
        create_risk_factor_chart(df)
    )

    combined_risk_chart = (
        create_combined_risk_chart(df)
    )

    employment_analysis = (
        make_category_analysis(
            df,
            "Employment_Status",
        )
    )

    loan_purpose_analysis = (
        make_category_analysis(
            df,
            "Loan_Purpose",
        )
    )

    # Important:
    # combined_risk_analysis can be a Pandas DataFrame.
    # Jinja cannot evaluate a DataFrame directly in:
    # {% if combined_risk_analysis %}
    combined_risk_analysis = (
        template_value(
            combined_risk_analysis
        )
    )

    risk_summary = template_value(
        risk_summary
    )

    return render_template(
        "dashboard.html",
        kpis=kpis,
        credit_score_chart=credit_score_chart,
        dti_chart=dti_chart,
        income_chart=income_chart,
        loan_amount_chart=loan_amount_chart,
        employment_chart=employment_chart,
        loan_purpose_chart=loan_purpose_chart,
        employment_analysis=employment_analysis,
        loan_purpose_analysis=loan_purpose_analysis,
        risk_summary=risk_summary,
        combined_risk_analysis=combined_risk_analysis,
        risk_factor_chart=risk_factor_chart,
        combined_risk_chart=combined_risk_chart,
        insights=create_project_insights(df),
        risk_thresholds={},
    )


@app.route("/model")
def model_page():
    evaluation = evaluate_random_forest()

    return render_template(
        "model.html",
        evaluation=evaluation,
        metrics=evaluation,
    )


@app.route(
    "/personal",
    methods=["GET", "POST"],
)
def personal():
    if request.method == "POST":
        session["personal"] = {
            "Age": request.form["Age"],
            "Gender": request.form["Gender"],
            "Marital_Status": request.form[
                "Marital_Status"
            ],
            "Dependents": request.form[
                "Dependents"
            ],
            "Education_Level": request.form[
                "Education_Level"
            ],
        }

        return redirect(
            url_for("employment")
        )

    return render_template(
        "personal.html",
        personal=session.get(
            "personal",
            {},
        ),
    )


@app.route(
    "/employment",
    methods=["GET", "POST"],
)
def employment():
    if "personal" not in session:
        return redirect(
            url_for("personal")
        )

    if request.method == "POST":
        session["employment"] = {
            "Employment_Status": request.form[
                "Employment_Status"
            ],
            "Employer_Category": request.form[
                "Employer_Category"
            ],
        }

        return redirect(
            url_for("financial")
        )

    return render_template(
        "employment.html",
        employment=session.get(
            "employment",
            {},
        ),
    )


@app.route(
    "/financial",
    methods=["GET", "POST"],
)
def financial():
    if "employment" not in session:
        return redirect(
            url_for("employment")
        )

    if request.method == "POST":
        session["financial"] = {
            "Applicant_Income": request.form[
                "Applicant_Income"
            ],
            "Coapplicant_Income": request.form[
                "Coapplicant_Income"
            ],
            "Savings": request.form["Savings"],
            "Existing_Loans": request.form[
                "Existing_Loans"
            ],
            "Credit_Score": request.form[
                "Credit_Score"
            ],
            "DTI_Ratio": request.form[
                "DTI_Ratio"
            ],
        }

        return redirect(
            url_for("loan")
        )

    return render_template(
        "financial.html",
        financial=session.get(
            "financial",
            {},
        ),
    )


@app.route(
    "/loan",
    methods=["GET", "POST"],
)
def loan():
    if "financial" not in session:
        return redirect(
            url_for("financial")
        )

    if request.method == "POST":
        session["loan"] = {
            "Loan_Amount": request.form[
                "Loan_Amount"
            ],
            "Loan_Term": request.form[
                "Loan_Term"
            ],
            "Loan_Purpose": request.form[
                "Loan_Purpose"
            ],
            "Property_Area": request.form[
                "Property_Area"
            ],
            "Collateral_Value": request.form[
                "Collateral_Value"
            ],
        }

        return redirect(
            url_for("predict")
        )

    return render_template(
        "loan.html",
        loan=session.get(
            "loan",
            {},
        ),
    )


@app.route(
    "/predict",
    methods=["GET"],
)
def predict():
    required_sections = [
        "personal",
        "employment",
        "financial",
        "loan",
    ]

    for section in required_sections:
        if section not in session:
            return redirect(
                url_for(section)
            )

    input_data = {}

    input_data.update(
        session["personal"]
    )

    input_data.update(
        session["employment"]
    )

    input_data.update(
        session["financial"]
    )

    input_data.update(
        session["loan"]
    )

    input_data = normalize_applicant(
        input_data
    )

    try:
        prediction = predict_loan(
            input_data
        )

        if str(prediction).strip().lower() == "yes":
            message = (
                "Based on the provided information, "
                "the model predicts that the loan "
                "may be approved."
            )
        else:
            message = (
                "Based on the provided information, "
                "the model predicts that the loan "
                "may not be approved."
            )

        session.clear()

        return render_template(
            "result.html",
            prediction=prediction,
            result=prediction,
            message=message,
            applicant=input_data,
        )

    except Exception as error:
        print("\n" + "=" * 70)
        print("LOANLENS PREDICTION ERROR")
        print("=" * 70)
        print(
            f"Error Type: {type(error).__name__}"
        )
        print(
            f"Error Message: {error}"
        )
        traceback.print_exc()
        print("=" * 70 + "\n")

        return render_template(
            "result.html",
            prediction="Unable to generate prediction",
            result="Unable to generate prediction",
            message=(
                "The prediction could not be generated. "
                "Check the terminal for the exact error."
            ),
            applicant=input_data,
            error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )


@app.route(
    "/api/predict",
    methods=["POST"],
)
def api_predict():
    try:
        data = request.get_json(
            silent=True
        )

        if not data:
            return jsonify(
                {
                    "success": False,
                    "error": "No input data received.",
                }
            ), 400

        data = normalize_applicant(
            data
        )

        prediction = predict_loan(
            data
        )

        return jsonify(
            {
                "success": True,
                "prediction": prediction,
            }
        )

    except Exception as error:
        print("\n" + "=" * 70)
        print("LOANLENS API PREDICTION ERROR")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70 + "\n")

        return jsonify(
            {
                "success": False,
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }
        ), 500


@app.route("/clear")
def clear():
    session.clear()

    return redirect(
        url_for("dashboard")
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
