# Utah Gymnastics Analytics Dashboard

## Overview
This repository contains an **anonymized sports performance dashboard** built with **Streamlit**, **Plotly**, and **Python**. It demonstrates data cleaning, visualization, and statistical modeling in an interactive format, originally developed during a summer internship with Utah Athletics.

## Features
- Athlete‑level and team‑level performance views  
- Fatigue flagging based on RSI z‑scores  
- Regression analysis linking jump metrics to meet scores  
- Season‑over‑season trend comparisons  
- Fully anonymized athlete identifiers  

## Technologies
- **Streamlit** for interactive web app deployment  
- **Plotly** for dynamic visualizations  
- **Pandas / NumPy / SciPy / StatsModels** for data analysis  

## Files
| File | Purpose |
|------|----------|
| `jump_dashboard_anon.py` | Main Streamlit app |
| `requirements.txt` | Python dependencies |
| `finaldf.pkl` | Meet performance data (anonymized) |
| `fulljumpdf.csv` | Jump test data (anonymized) |
| `ActiveGym_Jumps.xlsx` | Source jump data across seasons |

## Deployment
To run locally:
```bash
pip install -r requirements.txt
streamlit run jump_dashboard_anon.py
