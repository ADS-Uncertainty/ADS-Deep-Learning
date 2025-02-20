import pandas as pd
import numpy as np

def pearson_correlation(line1, line2):
    # Pandas Series
    series1 = pd.Series(line1)
    series2 = pd.Series(line2)

    correlation = series1.corr(series2, method='pearson')

    return correlation

# Replace it with your results
# line1_data = [0.689,0.801,0.885,0.94,0.984,1.022,1.059,1.097,1.131,1.172,1.212]    # MCD
line1_data = [1.569,1.649,1.773,1.9,2.022,2.132,2.234,2.328,2.419,2.508,2.597]    # DEs

# line2_data = [0.594,0.611,0.632,0.649,0.662,0.674,0.682,0.688,0.693,0.697,0.7]    # TET
# line2_data = [0.858,0.888,0.922,0.951,0.974,0.993,1.007,1.018,1.028,1.035,1.041]  # TIT
line2_data = [0.01101,0.01152,0.01221,0.01282,0.01333,0.01381,0.01416,0.0145,0.01479,0.01501,0.01523]  # CPI

# Pearson correlation coefficient Calculation
correlation_coefficient = pearson_correlation(line1_data, line2_data)
print("Pearson correlation coefficient：", correlation_coefficient)