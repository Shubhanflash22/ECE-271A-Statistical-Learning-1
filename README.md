# ECE-271A-Statistical-Learing-1
A machine learning project that predicts the physical activity of smartwatch users based on sensor data. This project demonstrates data aggregation, preprocessing, and predictive modeling for real-world activity recognition.

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Model](#model)
* [Results](#results)
* [Future Work](#future-work)
* [License](#license)

## Project Overview

This project collects data from smartwatches worn by 30 users and predicts their physical activity using machine learning techniques. The goal is to recognize activities such as walking, running, sitting, or other movements in real-time scenarios.

Key steps include:

* Aggregating large datasets from multiple users.
* Preprocessing and cleaning noisy sensor data.
* Analyzing the data to extract meaningful features.
* Training machine learning models to predict user activity accurately.

## Dataset

* Data collected from ~30 smartwatch users.
* Includes parameters like accelerometer, gyroscope, heart rate, and other motion sensors.
* Cleaned and preprocessed to remove noise and inconsistencies.

## Features

* **Data Aggregation:** Combine multi-user smartwatch data into a unified dataset.
* **Data Preprocessing:** Handle missing values, normalize sensor readings, and extract relevant features.
* **Activity Prediction:** Model predicts the physical activity of the wearer in real-time scenarios.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/human-activity-smartwatch.git
cd human-activity-smartwatch

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the required format (CSV with sensor parameters).
2. Run the preprocessing script:

```bash
python preprocess_data.py
```

3. Train the machine learning model:

```bash
python train_model.py
```

4. Predict activities on new smartwatch data:

```bash
python predict_activity.py --input new_data.csv
```

## Model

* Uses classical machine learning algorithms for activity recognition (e.g., Random Forest, SVM).
* Input features include sensor readings like acceleration, gyroscope data, and heart rate.
* Output: Predicted activity label for each timestamp.

## Results

* Achieved high accuracy when tested on real-world smartwatch data.
* Model can reliably predict activities such as walking, running, sitting, and more.

## Future Work

* Expand dataset with more users for better generalization.
* Explore deep learning models (e.g., LSTM, CNN) for temporal sensor data.
* Deploy as a real-time smartwatch application for fitness tracking.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
