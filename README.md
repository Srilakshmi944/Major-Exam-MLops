# Major-Exam-MLops

# MLOps Linear Regression Assignment

This project implements a complete MLOps pipeline using a Linear Regression model trained on the California Housing dataset. The goal is to automate the ML lifecycle including training, evaluation, quantization, containerization using Docker, and continuous integration using GitHub Actions.

## 📊 Model Overview
- **Model**: `LinearRegression` from `scikit-learn`
- **Dataset**: California Housing (from `sklearn.datasets`)
- **Pipeline Components**:
  - Model training (`train.py`)
  - Testing (`test_train.py`)
  - Manual quantization (`quantize.py`)
  - Prediction (`predict.py`)
  - Dockerization (`Dockerfile`)
  - CI/CD using GitHub Actions (`.github/workflows/ci.yml`)

## 🧪 Evaluation and Quantization
To optimize model size without significantly affecting performance, we performed manual quantization by converting model coefficients to `float16`.

### ✅ Comparison Table

| Metric                     | Before Quantization | After Quantization |
|---------------------------|----------------------|---------------------|
| R² Score                  | 0.575787             | 0.575467            |
| Model Size (KB)           | 0.68 KB              | 0.32 KB             |

> ℹ️ Note: These values are based on sample outputs and may vary slightly with each training run.

## 📁 Project Structure
```
mlops-linear-regression/
├── .github/workflows/ci.yml        # GitHub Actions for CI/CD
├── src/
│   ├── train.py                    # Model training
│   ├── predict.py                 # Run predictions
│   ├── quantize.py                # Quantization logic
│   └── utils.py                   # Shared utility functions
├── tests/
│   └── test_train.py              # Unit tests
├── Dockerfile                     # Docker container config
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
└── .gitignore                     # Files to ignore in version control
```

## 🛠 Setup Instructions
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py

# Run quantization
python src/quantize.py

# Run tests
pytest

# Build and run Docker container
docker build -t mlops-model .
docker run --rm mlops-model
```

## 🚀 CI/CD
GitHub Actions are configured to:
- Run unit tests
- Train and quantize model
- Build and test Docker container

