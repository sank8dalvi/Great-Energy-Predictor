from flask import Flask, render_template, url_for, request
from predictor import run
app = Flask(__name__)
@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/loading', methods=['POST'])
def loadLoadingPage():
    lab = [ 0,  4,  1,  9,  6, 15, 11,  2, 10,  8,  3, 14,  7, 13,  5, 12]
    cols = ['Education', 'Lodging/residential', 'Office',
        'Entertainment/public assembly', 'Other', 'Retail', 'Parking',
        'Public services', 'Warehouse/storage', 'Food sales and service',
        'Religious worship', 'Healthcare', 'Utility', 'Technology/science',
        'Manufacturing/industrial', 'Services']
    label = {cols[i]:lab[i] for i in range(len(cols))}
    print("Start predictions........")
    ip = ['107','107','0','2016-01-01 00:00:00','3',label['Education'],'97532','2005','10','3.8','255','2.4','1','1020.9','240','3.1']
    currentPrediction = run(ip)
    print(currentPrediction)
    return render_template('loading.html', prediction = str(currentPrediction))

app.run(threaded=False, debug=True)