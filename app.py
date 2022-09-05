from datetime import timedelta
import datetime
from ssl import OP_NO_TLSv1_1
from flask import Flask, render_template, redirect, request
from flask.helpers import flash
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
import warnings

from werkzeug import debug
warnings.filterwarnings('ignore')
from random import randint
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
from flask_socketio import SocketIO
from datetime import timedelta, date


from tempfile import TemporaryDirectory

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

hist_df = pd.read_csv('eur_data_inr.csv')
start_date = date(2022, 8, 28)
end_date = datetime.date.today()
df = pd.DataFrame()
for single_date in daterange(start_date, end_date):
    dfs = pd.read_html(f'https://www.xe.com/currencytables/?from=EUR&date={single_date.strftime("%Y-%m-%d")}')[0]
    dfs['Date'] = single_date.strftime("%Y-%m-%d")
    df = df.append(dfs)
df.to_csv('eur_data.csv')
idr_df = df[df['Currency'] == 'IDR']
#idr_df.pop('Rate')
#idr_df.pop("Change")
idr_df.head(5)
idr_df = pd.concat([hist_df, idr_df], ignore_index=True)      

length = len(idr_df)
data_day1=idr_df[length-1:]
data_day2=idr_df[length-2:length-1]
data_day7=idr_df[length-7:length-6]
data_day15=idr_df[length-15:length-14]
data_day100=idr_df[length-100:length-99]

change_1= float(data_day2['Units per EUR'])-float(data_day1['Units per EUR'])
change_7=float(data_day7['Units per EUR'])-float(data_day1['Units per EUR'])
change_15=float(data_day15['Units per EUR'])-float(data_day1['Units per EUR'])
change_100=float(data_day100['Units per EUR'])-float(data_day1['Units per EUR'])

price_day1=float(data_day1['Units per EUR'])

import os
app = Flask("_name_")
app.config["IMAGE_UPLOADS"] = "static/img/"
app.config["Graph_UPLOADS"] = "static/graph/"
socketio=SocketIO(app)
@app.route('/')
def index():

    actual_chart = go.Scatter (x=idr_df["Date"], y=idr_df["Units per EUR"], name='Data')

    with TemporaryDirectory() as tmp_dir:
        filename = tmp_dir + "tmp.html"
        py.plot([actual_chart],filename = filename, auto_open=False)
        with open(filename, "r") as f:
            graph_html = f.read()

        IS_FORECAST = False
        return render_template("index.html", price_day1=price_day1, change_1=change_1, change_7=change_7, change_15=change_15, change_100=change_100, graph_html=graph_html, IS_FORECAST=IS_FORECAST)


@app.route('/submit',methods=['POST'])
def submit_data():
    try:
        s2=int(request.form['parameter'])
        s1=request.form['options']
    except:
        flash("Please provide valid inputs")
        return redirect("/")

    df = pd.read_csv('eur_data_inr.csv', usecols=["Date", "Units per EUR"])
    df['Date']= pd.to_datetime(df.Date)

    x=np.array(range(88))
    x=x.reshape(-1,1)

    df['day']=x+1
    x=df.drop(['Date','Units per EUR'],axis = 1)
    y=df['Units per EUR']
    features = x.columns

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model10_svr = SVR(kernel='rbf', C=100 ,epsilon=1, gamma=0.01)
    model10_svr.fit(x_train,y_train)

    oot = pd.DataFrame(pd.date_range('2022-9-5', periods=s2, freq=s1))
    oot.rename(columns={0:'Date'}, inplace=True)
    j = 1
    for i in range(len(oot)) :
        oot.loc[i, 'day'] = len(x)+j
        j=j+i
    
    new_data = df[:-20].append(oot)

    for i in range(len(oot)) :
        new_data.loc[i, 'Units per EUR'] = model10_svr.predict(new_data[new_data.index==i][features])
    
    oot_new=new_data
    oot_new[['Units per EUR', 'Date']].tail()
    
   
    forecast_data_orig = oot_new
    #forecast_data_orig['Units per EUR']=np.exp(forecast_data_orig['Units per EUR'])
    #forecast_data_orig['Date']=np.exp(forecast_data_orig['Date'])
    final_df=pd.DataFrame(forecast_data_orig)



    data = idr_df.set_index(['Date'])
    data=data[['Units per EUR']]
    data.columns=['Actual']
    pred=oot_new.set_index(['Date'])
    pred=pred[['Units per EUR']]
    pred.columns=['Prediction']

    #actual_chart = go.Scatter(y=idr_df["Units per EUR"], x=idr_df['Date'], name='Actual', legendgroup="Actual",line=dict(color='blue'))
    predict_chart = go.Scatter(y=s2.append, x=oot_new['Date'], name='predicted',legendgroup="predicted",line=dict(color='red'), showlegend=True)

    #chart = go.Figure(actual_chart + predict_chart)
    #predict_chart = go.Scatter(y=oot_new["Units per EUR"], x=oot_new['Date'], name='predicted')



    with TemporaryDirectory() as tmp_dir:
        filename= tmp_dir + "tmp.html"
        #df_all=pd.merge(data, pred, how = 'outer', left_index=True, right_index=True)
        py.plot([predict_chart], filename = filename, auto_open=False)
        with open(filename, "r") as f :
            graph_html= f.read()
    if s1=="D":
        value="Days"
    elif s1=="M":
        value="Month"
    else:
        value="Year"

    final_df_1=final_df[[ 'Units per EUR', 'Date']].tail(s2)
    final_df_1=final_df_1.rename(columns={'Units per Euro': 'Prediksi Nilai Kurs', 'Date': 'Tanggal'})
    final_df_1.reset_index(drop=True, inplace=True)
    IS_FORECAST = True

    table = final_df_1.to_html(classes='table table-striped', border=0)
    table = table.replace('tr style="text-align: right;"', 'tr style="text-align: center;"')
    table = table.replace('<th></th>', '')
    table = table.replace('<th>', '<th colspan="2">', 1)
    print(table)
    return render_template("index.html",price_day1=price_day1,change_1=change_1,change_7=change_7,change_15=change_15,change_100=change_100, graph_html=graph_html, parameter=s2,table=table, IS_FORECAST = IS_FORECAST)


   

    
if __name__ =="__main__":


    socketio.run(app, port=8000, debug=True)
