from flask import Flask, render_template, request
import requests
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

API_KEY = os.getenv('API_KEY')
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

@app.route('/', methods=['GET', 'POST'])
def index():
    weather_data = None
    error = None
    if request.method == 'POST':
        city = request.form['city']  # <-- city is defined here
        params = {
            'q': city,
            'appid': API_KEY,
            'units': 'metric'
        }
        try:
            response = requests.get(BASE_URL, params=params)
            if response.status_code == 200:
                weather_data = response.json()
                icon_code = weather_data['weather'][0]['icon']
                weather_data['icon_url'] = f"static/icons/{icon_code}.png"
            else:
                error = "City not found or API error. Please check the name and try again."
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    return render_template('index.html', weather=weather_data, error=error)

if __name__ == '__main__':
    app.run(debug=True)
