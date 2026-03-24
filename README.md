# 🌊 River Flow Prediction | Artificial Neural Networks Coursework

This repository contains a custom-built **Multi-Layer Perceptron (MLP)** designed to predict the mean daily river flow at **Skelton**, North Yorkshire. This project was developed as part of the AI Methods module, focusing on time-series forecasting using environmental data.

---

### 📊 **Project Overview**
The objective is to predict river flow one day in advance by analyzing historical rainfall and flow data from multiple gauging stations. The model evaluates the correlation between upstream stations (like Malham Tarn and Arkengarthdale) and the flow at Skelton.

**Key Features:**
* **Custom MLP Architecture:** Implemented with backpropagation, momentum-based updates, and weight decay for optimized learning.
* **Advanced Pre-processing:** * Outlier removal using the **IQR method** (Threshold: $Q1 - 10 \times IQR$ to $Q3 + 10 \times IQR$).
    * Data standardization to a range of **[0.1, 0.9]** for neural network compatibility.
    * Feature encoding for seasonal data (Month_sin/Month_cos).
* **Data Split:** Utilizes a standard **60/20/20** split (Training: 1100, Validation: 137, Test: 138 entries).
* **Comparative Analysis:** Performance is benchmarked against a **Linear Regression Baseline** to validate the MLP's ability to handle non-linear connections.

---

### 🛠 **Tech Stack**
* **Language:** Python
* **Numerical Computing:** `NumPy`
* **Data Manipulation:** `Pandas`
* **Machine Learning:** `Scikit-learn` (Baseline only)
* **Visualization:** `Matplotlib`, `Seaborn`

---

### 📂 **Project Structure**
* **`weather.py`**: The core implementation including the MLP class, data cleaning, and training loop.
* **`Ouse93-96 - Student.xlsx`**: The primary dataset (Flow and Rainfall data 1993-1996).
* **`header.cpp`**: C++ supplementary logic for data handling.
* **`.gitignore`**: Configured to exclude local environment files and heavy binaries.

---

### 🚀 **Getting Started**

#### **1. Installation**
Clone the repository and install the necessary Python environment:
```bash
git clone [https://github.com/tolufeko/weather-flow-prediction.git](https://github.com/tolufeko/weather-flow-prediction.git)
cd "ai methods"
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
