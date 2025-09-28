from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

## Route for a home page

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template("home.html")
    else:
        data = CustomData(
                    item_weight = float(request.form.get('item_weight')),
                    item_visibility = float(request.form.get('item_visibility')),
                    item_mrp = float(request.form.get('item_mrp')),
                    item_type = request.form.get('item_type'),
                    item_fat_content = request.form.get('item_fat_content'),
                    outlet_identifier = request.form.get('outlet_identifier'),
                    outlet_establishment_year = int(request.form.get('outlet_establishment_year')),
                    outlet_type = request.form.get('outlet_type'),
                    outlet_size = request.form.get('outlet_size'),
                    outlet_location_type = request.form.get('outlet_location_type')
        )
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print(results)
        if results:
            return render_template("home.html",result=results[0])
        else:
            return render_template("home.html", result="Error in prediction")
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)