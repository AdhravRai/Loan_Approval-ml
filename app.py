from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

application = Flask(__name__)

app = application


# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Prediction Route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':

        return render_template('home.html')

    else:

        data = CustomData(

            Gender=request.form.get('Gender'),

            Married=request.form.get('Married'),

            Dependents=request.form.get('Dependents'),

            Education=request.form.get('Education'),

            Self_Employed=request.form.get('Self_Employed'),

            ApplicantIncome=float(
                request.form.get('ApplicantIncome')
            ),

            CoapplicantIncome=float(
                request.form.get('CoapplicantIncome')
            ),

            LoanAmount=float(
                request.form.get('LoanAmount')
            ),

            Loan_Amount_Term=float(
                request.form.get('Loan_Amount_Term')
            ),

            Credit_History=float(
                request.form.get('Credit_History')
            ),

            Property_Area=request.form.get('Property_Area')
        )

        pred_df = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()

        result = predict_pipeline.predict(pred_df)

        if result == 1:
            prediction = "Loan Approved"
        else:
            prediction = "Loan Rejected"

        return render_template(
            'home.html',
            results=prediction
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)