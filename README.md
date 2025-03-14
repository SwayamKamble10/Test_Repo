# Algerian Forest Fire Prediction

This project implements a machine learning model to predict forest fire probabilities in Algeria based on various environmental parameters. The application uses a Ridge Regression model trained on the Algerian Forest Fires dataset.

## Features

- Web-based interface for input parameters
- Real-time prediction using Ridge Regression model
- Calculation of Fire Weather Index (FWI) and Build Up Index (BUI)
- Interactive results display with detailed parameter breakdown
- Bootstrap-based responsive design

## Required Parameters

- Temperature (°C)
- Relative Humidity (%)
- Wind Speed (km/h)
- Rain (mm)
- FFMC (Fine Fuel Moisture Code)
- DMC (Duff Moisture Code)
- DC (Drought Code)
- ISI (Initial Spread Index)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/SwayamKamble10/Test_Repo.git
   cd Test_Repo
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python application.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Model Details

The prediction model uses Ridge Regression trained on the Algerian Forest Fires dataset. It takes into account various environmental parameters and calculates additional indices (BUI and FWI) to provide accurate fire probability predictions.

## Technologies Used

- Python
- Flask
- Scikit-learn
- Bootstrap
- HTML/CSS
- Pandas
- NumPy

## Author

Swayam Kamble
