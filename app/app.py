from flask import Flask, request, redirect, render_template , session , url_for
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from flask_session import Session

import os
import pickle
import pandas as pd
import numpy as np





app = Flask(__name__, template_folder='templates')

app.secret_key = "Hello"
app.config["DATASET_UPLOADS"] = r"C:\Users\msi\Desktop\Flask\app\uploads"
app.config["ALLOWED_DATASET_EXTENSIONS"] = ["CSV"]
app.config["SECRET_KEY"] =  "SDSADfbdfb_dgfsbsfgb"
 



def allowed_dataset(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_DATASET_EXTENSIONS"]:
        return True
    else:
        return False




@app.route("/", methods=["GET", "POST"])
def upload_dataset():

    if request.method == "POST":
        if request.files:

            dataset = request.files["Dataset"]
            
            if dataset.filename == "":
                print("No filename")
                return redirect(request.url)



            if allowed_dataset(dataset.filename):
                filename = secure_filename(dataset.filename)
                dataset.save(os.path.join(app.config["DATASET_UPLOADS"], filename))
                print("dataset saved")

                df = pd.read_csv(f"uploads\{filename}")
                session['file_name'] = filename
                n_columns = df.shape[1]
                features_list = []
                for i in range(0 , n_columns-1):
                    features_list.append(df.columns[i])
                X = df[features_list]
                y = df[df.columns[n_columns-1]]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test= sc.transform(X_test)


                if request.form.get("models") =="one":
                    classifier = RandomForestClassifier()
                if request.form.get("models") =="two":
                    classifier = GaussianNB()
                if request.form.get("models") =="three":
                    classifier = KNeighborsClassifier()
                if request.form.get("models") =="four":
                    classifier = LogisticRegression()
                if request.form.get("models") =="five":
                    classifier = SVC()



                classifier.fit(X_train, y_train)
                pickle.dump(classifier, open("model.pkl", "wb"))
                return redirect(url_for('predict'))

            else:

                print("That file extension is not allowed")
                return redirect(request.url)


    return render_template("upload_dataset.html")




@app.route("/predict", methods = ["POST" , "GET"])
def predict():

    filename = session.get('file_name' , None)
    df = pd.read_csv(f"uploads\{filename}")
    n_columns = df.shape[1]
    features_list = []
    for i in range(0 , n_columns-1):
        features_list.append(df.columns[i])
    length = len(features_list)

    if request.method == "POST":

        model = pickle.load(open("model.pkl", "rb"))
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)
        print(features_list)

        return render_template("index.html" , prediction_text =  f"Model Prediction : {prediction}" , length = length , features_list = features_list)
    return render_template("index.html" ,  length = length , features_list = features_list)




if __name__ == '__main__':
    app.run(debug = True)




