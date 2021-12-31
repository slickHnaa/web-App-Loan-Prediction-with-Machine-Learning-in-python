import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'home ownership':2,
                            'Purpose':,
                            'long term':0,
                            'short term':1,
                            'current loan amount':12232.0,
                            'credit score':716.612735,
                            'years in current job':1.0, 
                            'annual income':46643.0,
                            'Monthly Dedt',
                            'years of credit history':18.0,
                            'number of open accounts':12.0, 
                            'Current Credit Balance':,
                            'Maximum Open Credit':,})

print(r.json())