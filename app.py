from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

app = Flask(__name__)

# Load dataset and preprocess (same as your code)
df = pd.read_excel(r"Data.xlsx")
df.dropna(inplace=True)

# Feature Engineering
df['Journey_day'] = pd.to_datetime(df.Date_of_Journey, format='%d/%m/%Y').dt.day
df['Journey_month'] = pd.to_datetime(df.Date_of_Journey, format='%d/%m/%Y').dt.month
df['Arrival_hour'] = pd.to_datetime(df.Arrival_Time).dt.hour
df['Arrival_min'] = pd.to_datetime(df.Arrival_Time).dt.minute
df['Dep_hour'] = pd.to_datetime(df.Dep_Time).dt.hour
df['Dep_min'] = pd.to_datetime(df.Dep_Time).dt.minute
df.drop(columns=['Date_of_Journey', 'Arrival_Time', 'Dep_Time'], inplace=True)

# Processing Duration
duration = list(df.Duration)
duration_hours = []
duration_mins = []

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        elif 'm' in duration[i]:
            duration[i] = "0h {}".format(duration[i].strip())
    duration_hours.append(int(duration[i].split()[0][:-1]))
    duration_mins.append(int(duration[i].split()[1][:-1]))

df['Duration_hours'] = duration_hours
df['Duration_mins'] = duration_mins
df.drop(['Duration'], axis=1, inplace=True)

# Clean Airline data
df.Airline = df.Airline.apply(lambda x: x.strip())
airline_stats = df['Airline'].value_counts(ascending=False)
airline_stats_less_than_10 = airline_stats[airline_stats <= 10]
df.Airline = df.Airline.apply(lambda x: 'other' if x in airline_stats_less_than_10 else x)

# Drop unnecessary columns
df.drop(columns=['Route', 'Additional_Info'], inplace=True)
df.replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}, inplace=True)

# One-Hot Encoding
dfdummies = pd.get_dummies(data=df, columns=['Airline', 'Source', 'Destination'], drop_first=True)

# Splitting the data
x = dfdummies.drop('Price', axis=1)
y = dfdummies['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# Training the model
rf_model = RandomForestRegressor()
rf_model.fit(x_train, y_train)

# Function to predict flight price
def predict_flight_price(dep_date_time, arr_date_time, source, destination, stops, airline):
    # Parse the date-time inputs in 'YYYY-MM-DDTHH:MM' format from the HTML form
    dep_datetime = datetime.strptime(dep_date_time, '%Y-%m-%dT%H:%M')
    arr_datetime = datetime.strptime(arr_date_time, '%Y-%m-%dT%H:%M')

    # Check if arrival time is before departure time
    if arr_datetime <= dep_datetime:
        return "Error: Arrival time cannot be before or the same as the departure time."

    # Prepare the input data for prediction
    input_data = {
        'Journey_day': dep_datetime.day,
        'Journey_month': dep_datetime.month,
        'Arrival_hour': arr_datetime.hour,
        'Arrival_min': arr_datetime.minute,
        'Dep_hour': dep_datetime.hour,
        'Dep_min': dep_datetime.minute,
        'Total_Stops': stops,
        'Duration_hours': (arr_datetime - dep_datetime).seconds // 3600,
        'Duration_mins': (arr_datetime - dep_datetime).seconds % 3600 // 60
    }

    # Airline, Source, and Destination Encoding
    for col in x_train.columns:
        if col.startswith('Airline_'):
            input_data[col] = 1 if airline == col.split('_')[1] else 0
        elif col.startswith('Source_'):
            input_data[col] = 1 if source == col.split('_')[1] else 0
        elif col.startswith('Destination_'):
            input_data[col] = 1 if destination == col.split('_')[1] else 0

    # Convert input_data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Handle missing columns
    missing_cols = set(x_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    # Reorder input_df to match the training data's column order
    input_df = input_df[x_train.columns]

    # Predict the price
    predicted_price = rf_model.predict(input_df)[0]

    return f"The predicted flight price is: ₹{predicted_price:.2f}"


# Home route (Form Page)
@app.route('/')
def index():
    return render_template('index.html')

# Result route (Handle Form Submission)
@app.route('/predict', methods=['POST'])
def predict():
    dep_date_time = request.form['dep_date_time']
    arr_date_time = request.form['arr_date_time']
    source = request.form['source']
    destination = request.form['destination']
    stops = int(request.form['stops'])
    airline = request.form['airline']

    predicted_price = predict_flight_price(dep_date_time, arr_date_time, source, destination, stops, airline)

    return render_template('result.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
