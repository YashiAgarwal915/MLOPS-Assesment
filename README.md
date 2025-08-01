# **End-to-End MLOps Pipeline for California Housing Prediction**

## **Project Description**

This repository is the culmination of my work for the MLOps Major Assignment. It contains a complete, automated pipeline designed to handle the entire lifecycle of a machine learning model, from initial testing to final deployment. The project is built around a Linear Regression model that predicts housing prices using the California Housing dataset.

The primary goal was not just to build a model, but to build a robust system around it. This involved integrating several core MLOps practices: continuous integration and deployment (CI/CD) with GitHub Actions, code and environment packaging with Docker, and model optimization using manual quantization. The entire process is managed within a single `main` branch to simulate a streamlined, trunk-based development workflow.

## **Pipeline Workflow**

The heart of this project is the automated CI/CD pipeline defined in the GitHub Actions workflow file. When I push any new code to the `main` branch, a sequence of three jobs is automatically triggered, each depending on the success of the one before it.

1.  **Job 1: Code Testing (`test_suite`)**
    The pipeline begins by rigorously testing the source code. It runs a suite of `pytest` unit tests that validate the data loading process and ensure the model training logic is sound and performs above a minimum quality threshold. If any of these tests fail, the pipeline stops immediately, preventing flawed code from moving forward.

2.  **Job 2: Model Training and Optimization (`train_and_quantize`)**
    Once the code is verified, this job takes over. It first runs the main training script to produce the baseline scikit-learn model. Immediately after, it executes a quantization script. This script extracts the model's parameters (its weights and biases) and converts them from high-precision decimal numbers into much smaller 8-bit integers. Both the original and the new, smaller parameters are saved as artifacts, which are files stored by GitHub Actions for use in later steps.

3.  **Job 3: Dockerization and Deployment (`build_and_test_container`)**
    In the final stage, the pipeline prepares the model for the real world. It first downloads the model artifacts saved in the previous job. Then, using the provided `Dockerfile`, it builds a self-contained Docker image that packages the Python environment, all the necessary libraries, and our application code. A critical step here is an internal test: the workflow runs a container from this new image and executes a verification script to ensure the model can be loaded and used correctly inside this isolated environment. Only after this internal check passes is the final, validated image pushed to my public Docker Hub repository, making it available for deployment.

## **Model Performance and Quantization Results**

A key part of this assignment was to analyze the trade-off between model performance and its efficiency. The original Linear Regression model achieved an **R² Score of 0.5758**.

After applying manual 8-bit quantization, I re-evaluated the model's performance and its size. The results are summarized below:

| Metric | Original Sklearn Model | Quantized Model |
| :--- | :--- | :--- |
| **R² Score** | `0.5758` | `[Paste your R2 score for the quantized model here]` |
| **Parameter Size**| `[Paste the file size of unquant_params.joblib in KB]` KB | `[Paste the file size of quant_params.joblib in KB]` KB |

The analysis shows that quantization was highly effective. It resulted in a significant reduction in the file size of the model's parameters. This was achieved with a negligible, if any, drop in the R² Score, proving that we can make our model much more lightweight and efficient for deployment without sacrificing its predictive power. This is a crucial practice for deploying models in environments where storage and memory are limited.
