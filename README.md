# Titanic Survival Predictor: Docker Lab

A modified version of [Docker Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Docker_Labs/Lab2) from the MLOps course repo. A TensorFlow model is trained to predict whether a Titanic passenger survived, then served through a Flask web app, using either a multi-stage Dockerfile or Docker Compose.

## Changes from the original lab

### Dataset and model
- Replaced the Iris dataset with the **Titanic dataset** (`data/titanic.csv`, 891 passengers)
- Uses 4 features: passenger class, sex, age and fare. Rows with missing values are dropped, and sex is encoded as 0/1
- Changed the model from 3-class (`softmax`) to **binary classification** (`sigmoid`, `binary_crossentropy`)
- **Fixed a bug in the original:** the `StandardScaler` was fitted during training but never saved, so the serving app made predictions on unscaled input. The scaler is now saved as `scaler.pkl` and applied in `main.py`

### Flask app
- `main.py` returns the prediction **and the survival probability**, e.g. `{"predicted_class": "Survived", "survival_probability": 0.99}`
- Redesigned the form in `predict.html` with dropdowns for class and sex and number inputs for age and fare
- Removed unused code (`api_url`) and the Iris images (`statics/`)
- `requirements.txt`: added `pandas` and `joblib` and removed the unused `requests`

### Dockerfile
- **Dependencies are installed once:** a shared `base` stage installs the requirements, and both the training and serving stages build on it. The original ran `pip install` in each stage, which downloaded TensorFlow twice
- Copies `data/` into the training stage
- Copies both `my_model.keras` and `scaler.pkl` from the training stage into the serving stage
- **Fixed the port:** `EXPOSE 80` became `EXPOSE 4000` to match the port Flask actually runs on
- Removed the outdated `--trusted-host pypi.python.org` flag

### Docker Compose
- **Builds from the Dockerfile's `base` stage** instead of `python:3.10-slim`, so containers no longer run `pip install` on every start
- Mounts `data/` into the training container
- Passes both the model and the scaler through the shared `model_exchange` volume
- **Fixed the port mapping:** `80:80` became `4000:4000`
- Removed mounts for folders that no longer exist (`statics/`) or aren't needed (`templates/`)

## Two approaches

| | Dockerfile | Docker Compose |
|---|---|---|
| When the model is trained | At **build time**, baked into the image | At **run time**, in a separate container |
| How the model reaches serving | `COPY --from=model_training` | Shared volume `model_exchange` |
| Code changes | Need a rebuild | `src/` is mounted, so just restart |
| Best for | Shipping or deploying one image | Local development, multi-container setups |

## How to run

### Option 1: Dockerfile
```sh
docker build -t titanic-app .
docker run -p 4000:4000 titanic-app
```

### Option 2: Docker Compose
```sh
docker compose up --build
# stop with Ctrl+C, then:
docker compose down
```

Then open **http://localhost:4000/predict**.

## Example predictions

| Class | Sex | Age | Fare | Result |
|---|---|---|---|---|
| 1st | Female | 30 | £80 | Survived (99%) |
| 3rd | Male | 25 | £7.50 | Did not survive (14%) |
| 2nd | Male | 5 | £20 | Survived (57%) |

## Project structure
```
.
├── data/
│   └── titanic.csv
├── src/
│   ├── model_training.py
│   ├── main.py
│   └── templates/
│       └── predict.html
├── dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```
