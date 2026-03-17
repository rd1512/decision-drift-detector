# Decision Drift Detector

A production-style MLOps system that detects **silent changes in model decision behavior** over time — even when traditional performance metrics remain stable.


## Problem Statement

Most ML monitoring focuses on metrics such as accuracy, precision, or recall.  
In real-world deployments, however, a model can change *how it makes decisions* while maintaining similar overall performance.

These hidden shifts can lead to degraded user experience, fairness issues, or unexpected failures.

This project detects **decision drift** — changes in the distribution of model predictions — and alerts teams before problems escalate.


## Objectives

- Continuously log model predictions in production  
- Establish a baseline of normal decision behavior  
- Compare recent predictions against the baseline  
- Detect drift using statistical divergence  
- Trigger alerts when significant changes occur  
- Provide monitoring visibility through metrics  


## Key Concepts

- Decision Drift  
- Distribution Shift  
- Concept Drift  
- KL Divergence  
- Production ML Monitoring  
- MLOps Reliability  


## System Architecture

The system simulates a real production ML pipeline:

1. **Inference Service** — Serves predictions via FastAPI  
2. **Prediction Logging** — Stores outputs in a database  
3. **Baseline Builder** — Captures normal decision distribution  
4. **Drift Detector** — Compares baseline vs. recent predictions  
5. **Monitoring & Alerts** — Notifies when drift is detected  
6. **Traffic Simulator** — Generates synthetic production traffic  


## Tech Stack

- Python  
- FastAPI  
- Scikit-learn  
- SQLite  
- Prometheus  
- Slack Webhooks  
- Statistical Drift Detection  


## How Drift Detection Works

1. Collect predictions during normal operation  
2. Build a baseline distribution of model outputs  
3. Continuously gather recent predictions  
4. Compute KL divergence between baseline and recent data  
5. Trigger an alert if divergence exceeds a threshold  
