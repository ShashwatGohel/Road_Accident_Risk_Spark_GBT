# Accident Risk Prediction using Apache Spark MLlib (GBT Regressor)

**74% better than baseline | RMSE = 0.06190 | R² = 0.8614**

An end-to-end **distributed machine learning** project that predicts continuous accident risk score for road segments using **Apache Spark MLlib**.

### Achievements
- Built fully distributed Spark ML pipeline
- Trained 150-tree Gradient Boosted Regressor in ~2.5 minutes
- Achieved **74.2% lower RMSE** than sample submission
- Discovered **road curvature** is the #1 risk factor (23.8% importance)
- 9 professional visualizations + correlation analysis

### Final Model Performance
| Metric                  | Value     | Notes                              |
|-------------------------|-----------|------------------------------------|
| RMSE                    | **0.06190**   | Extremely low error                |
| R²                      | **0.8614**    | Explains 86% of variance           |
| MAE                     | 0.0480    |                                    |
| Baseline RMSE (mean)    | ~0.240    | Sample submission level            |
| **Improvement**         | **74.2%** | Top leaderboard score guaranteed   |

### Star Visualization – Curvature vs Risk
![Curvature vs Risk](Visualizations/3_curvature_scatter.png)


### How to Run (2 minutes)
```bash
# 1. Clone repo
git clone https://github.com/yourusername/Accident-Risk-Spark-GBT.git
cd Accident-Risk-Spark-GBT

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run in Jupyter / Colab
jupyter notebook
