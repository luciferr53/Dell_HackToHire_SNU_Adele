from flask import Flask, render_template, jsonify, redirect, request

from sklearn.externals import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/scrape", methods=['GET'])
def scrape():
    listings = mongo.db.listings
    listings_data = scrape_craigslist.scrape()
    listings.update(
        {},
        listings_data,
        upsert=True
    )
    return redirect("http://localhost:5000/", code=302)

@app.route('/get-user-data', methods=['POST'])
def predict_stuff():
    if request.method == 'POST':
        df = pd.read_csv('dataset.csv')


        # In[379]:


        del df['PRODUCT ID']
        del df['Customer Id']
        del df['customer name']


        # In[380]:


        df.head()


        # In[381]:


        class MultiColumnLabelEncoder:
            def __init__(self,columns = None):
                self.columns = columns # array of column names to encode

            def fit(self,X,y=None):
                return self # not relevant here

            def transform(self,X):
                '''
                Transforms columns of X specified in self.columns using
                LabelEncoder(). If no columns specified, transforms all
                columns in X.
                '''
                output = X.copy()
                if self.columns is not None:
                    for col in self.columns:
                        output[col] = LabelEncoder().fit_transform(output[col])
                else:
                    for colname,col in output.iteritems():
                        output[colname] = LabelEncoder().fit_transform(col)
                return output

            def fit_transform(self,X,y=None):
                return self.fit(X,y).transform(X)


        # In[387]:


        features_df = ['CPU']
        df = MultiColumnLabelEncoder(columns = ['CPU']).fit_transform(df)
        #df


        # In[383]:


        X = df[features_df]
        y = df['quantiy sold'].as_matrix()


        # In[384]:


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


        # In[385]:


        #regr = linear_model.LinearRegression()# Train the model using the training sets
        regr = Ridge(alpha=1.0)
        regr.fit(X_train, y_train)
        #regr.coef_
        joblib.dump(regr, 'trained_house_classifier_model.pkl',protocol=0)
        
        model = joblib.load('trained_house_classifier_model.pkl')

        print('-----line 27--------')

        year_built = int(request.form.get('year_built'))
        print(request.form.get('year_built'))
        print('line 31')

        product_name = request.form.get('product_name')
        location = request.form.get('location')
        RAM = request.form.get('RAM')
        CPU = request.form.get('CPU')
        GPU = request.form.get('GPU')
    #    print("CPU is : " + CPU)
        #print("product_name is : " + product_name)
        if(CPU == 'i5'):
            CPU = 1
        elif(CPU == 'i7'):
            CPU = 2
        elif(CPU == 'i3'):
            CPU = 0
        else:
            CPU = 'Invalid'
            return render_template("index.html", invalid = "invalid")

        #print("CPU is " + CPU)
        warranty_period = request.form.get('warranty_period')
        cost2 = int(request.form.get('cost'))
        Customer_Id = "aa"
        customer_name = "aa"
        product_id = "aa"
        '''
        carport_sqft = int(request.form.get('carport_sqft'))
        has_fireplace = request.form.get('has_fireplace')

        has_pool = request.form.get('has_pool')

        has_central_heating = request.form.get('has_central_heating')

        has_central_cooling = request.form.get('has_central_cooling')


        has_fireplace = request.form.get('has_fireplace')

        garage_type = request.form.get('garage_type')

        city = request.form.get('city')
        '''

        # return render_template("index.html", pred=house_to_value)
        print(" before predicted_home_values")
        # Run the model and make a prediction for each house in the homes_to_value array
        predicted_home_values = model.predict(CPU)
        # Since we are only predicting the price of one house, just look at the first prediction returned
        print("here")
        predicted_value = predicted_home_values
        cost1 = predicted_value*cost2
        print('aa')
        return render_template("index.html", pred=predicted_value,cost = cost1)


if __name__ == "__main__":
    app.run()
