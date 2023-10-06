import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load the trained model from the pickle file
model = pickle.load(open('model.pkl', 'rb'))

# Create a Flask app
app = Flask(__name__)

# Define a route to accept input and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from JSON request
        data = request.get_json()

        # Ensure the input data has the correct format
        if 'features' not in data:
            return jsonify({'error': 'Input data must contain a "features" key with an array of feature values'}), 400

        # Extract features from the input data
        features = data['features']

        # Check if the number of features matches the model's input shape
        if len(features) != len(model.feature_importances_):
            return jsonify({'error': 'Input data should contain the same number of features as the model expects'}), 400

        # Make predictions using the model
        prediction = model.predict([features])[0]

        # Map the predicted class index to the corresponding college name
        college_name = colleges[prediction]

        # Prepare the response
        response = {
            'prediction': college_name,
            'probability': max(model.predict_proba([features])[0])
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Define the mapping of class indexes to college names (as you've provided in your code)
    colleges = {
        0: 'Veermata Jijabai Technological Institute (VJTI)',
        1: 'Sardar Patel Institute of Technology (SPIT)',
        2: 'K. J. Somaiya College of Engineering',
        3: 'Thadomal Shahani Engineering College (TSEC)',
        4: 'Rajiv Gandhi Institute of Technology (RGIT)',
        5: 'Mukesh Patel School of Technology Management and Engineering (MPSTME)',
        6: 'Fr. Conceicao Rodrigues College of Engineering (CRCE)',
        7: 'Institute of Chemical Technology (ICT)',
        8: 'Sardar Patel College of Engineering (SPCE)',
        9: 'Dwarkadas J. Sanghvi College of Engineering (DJSCE)',
        10: 'Yadavrao Tasgaonkar Institute of Engineering & Technology (YTIET)',
        11: 'Shah & Anchor Kutchhi Engineering College (SAKEC)',
        12: "Vidyavardhini's College of Engineering and Technology",
        13: 'St. Francis Institute of Technology (SFIT)',
        14: 'SIES Graduate School of Technology (SIES GST)',
        15: "Pillai's HOC College of Engineering and Technology (PHCET)",
        16: 'Bharti Vidyapeeth College of Engineering (BVCOE)',
        17: 'Atharva College of Engineering (ACE)',
        18: 'Saraswati College of Engineering',
        19: 'Don Bosco Institute of Technology (DBIT)',
        20: 'Vidyalankar Institute of Technology (VIT)',
        21: 'Pillai College of Engineering (PCE)',
        22: 'Xavier Institute of Engineering (XIE)',
        23: 'Terna Engineering College',
        24: 'Datta Meghe College of Engineering',
        25: 'Lokmanya Tilak College of Engineering (LTCE)',
        26: 'Rizvi College of Engineering (RCOE)',
        27: 'Saraswati College of Engineering - Navi Mumbai',
        28: 'Bharti Vidyapeeth College of Engineering (BVCOE) - Navi Mumbai',
        29: 'St. Francis Institute of Technology (SFIT) - Mumbai',
        30: 'Atharva College of Engineering (ACE) - Mumbai',
        31: "Pillai's HOC College of Engineering and Technology (PHCET) - Raigad",
        32: 'Don Bosco Institute of Technology (DBIT) - Mumbai',
        33: 'Vidyalankar Institute of Technology (VIT) - Mumbai',
        34: 'Yadavrao Tasgaonkar Institute of Engineering & Technology (YTIET) - Raigad',
        35: 'Shah & Anchor Kutchhi Engineering College (SAKEC) - Mumbai',
        36: "Vidyavardhini's College of Engineering and Technology - Vasai",
        37: 'Rizvi College of Engineering (RCOE) - Mumbai',
        38: 'Xavier Institute of Engineering (XIE) - Mumbai',
        39: 'SIES Graduate School of Technology (SIES GST) - Navi Mumbai',
        40: 'Lokmanya Tilak College of Engineering (LTCE) - Mumbai',
        41: 'Terna Engineering College - Navi Mumbai',
        42: 'Datta Meghe College of Engineering - Navi Mumbai',
        43: 'K.J. Somaiya Institute of Engineering and Information Technology',
        44: 'Pillai College of Engineering (PCE) - Navi Mumbai',
        45: "Vivekanand Education Society's Institute of Technology (VESIT) - Mumbai",
        46: 'Institute of Chemical Technology (ICT) - Mumbai',
        47: 'Sardar Patel College of Engineering (SPCE) - Mumbai',
        48: 'K.J. Somaiya Institute of Engineering and Information Technology - Mumbai',
        49: 'Dwarkadas J. Sanghvi College of Engineering (DJSCE) - Mumbai',
        50: 'Indian Institute of Technology Bombay (IIT Bombay) - Mumbai'
    };

    # Run the Flask app on a specified port (e.g., 5000)
    app.run(debug=True, host='0.0.0.0')

