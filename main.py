from flask import Flask, request, jsonify, abort, json
# import sentimentAnalysis 
import numpy as np
import SecondAnalysis
from dotenv import load_dotenv
import os 

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
app.json_encoder = CustomEncoder

load_dotenv()
api_key = os.getenv('API_KEY')

def require_apikey(func):
    def wrapper(*args, **kwargs):
        if request.headers.get('X-API-KEY') == api_key:
            return func(*args, **kwargs)
        else:
            abort(401)  # Unauthorized access
    return wrapper

@app.route('/analyze', methods=['POST'])
@require_apikey 
def analyze(): 
    data = request.json 
    headlines = data['headlines'] 
    # pos_avg, neg_avg = sentimentAnalysis.analyzier(headlines)
    pos_avg, neg_avg = SecondAnalysis.getScore(headlines)

    return jsonify((pos_avg, neg_avg))
    # return jsonify({'positve_average': pos_avg, 'negative_avgerage': neg_avg})
    
if __name__ == '__main__':
    app.run(port=5000)
