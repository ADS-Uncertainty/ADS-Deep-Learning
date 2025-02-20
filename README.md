# Uncertainty Propagation from Sensor Data to Deep Learning Models in Autonomous Driving
## Overview

Deep learning has been widely used in Autonomous Driving Systems (ADS). Though significant progress has been made regarding their efficiency and accuracy, uncertainty remains a critical factor affecting ADS safety. We focus on the local path planning task in highway driving scenarios, which is a regression problem, and employ the LSTM-CSP model as the target model. We use Gaussian Processes to quantify the corresponding Aleatoric Uncertainty (AU). For Epistemic Uncertainty (EU), we employ two classic quantification methods, MC Dropout and Deep Ensembles. More importantly, we also studied the cost-effectiveness of MC Dropout and Deep Ensembles in selecting highly-uncertain samples for facilitating model retraining to improve prediction accuracy.

## Dependencies
* First step is to install the dependencies using requirements.txt file

  ```python
  pip install -r requirements.txt
  ```

## Resource
1. [Data](https://github.com/ADS-Uncertainty/ADS-Deep-Learning/tree/main/Data): We provide two test sets used in our experiments, including:
  * Testset
  * Car-following testset
2. [Models](https://github.com/ADS-Uncertainty/ADS-Deep-Learning/tree/main/Models): We provide 10 pre-trained LSTM-CSP models, including the LSTM-CSP-Optimal model.
3. [Pictures](https://github.com/ADS-Uncertainty/ADS-Deep-Learning/tree/main/Pic/Figure_5): The complete results of the overlap in highly-uncertain samples identified by MC Dropout and Deep Ensembles under different noise levels.

In our paper, we present only the results for the top 10% threshold at noise levels L2, L4, L6, and L8. The complete results can be found in 

## Usage
1. Download all the files to the local machine.
2. Run the programs in [Data Production](https://github.com/ADS-Uncertainty/ADS-Deep-Learning/tree/main/Program/Data%20Production) sequentially to generate the necessary data files.
3. Run the programs in [Baseline](https://github.com/ADS-Uncertainty/ADS-Deep-Learning/tree/main/Program/Baseline) to quantify the aleatoric and epistemic uncertainties of LSTM-CSP, and calculate the performance and safety metrics.

