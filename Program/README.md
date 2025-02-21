# Usage

## Data Production
1. Run the programs in [Data Production](https://github.com/ADS-Uncertainty/ADS-Deep-Learning/tree/main/Program/Data%20Production) sequentially to generate the necessary data files.
* `1_Fixed_noise_size.py`: Obtain the fixed random noise size used by MC Dropout and Deep Ensembles.
* `2_Fixed_noise_production.py`: Generate the fixed random noise into a folder, and directly apply it to the trajectory data when running the program.
* `3_Safety_data_production.py`: The calculation of Safety Metrics requires the re collection of data, as this type of data is fundamentally different from MC Dropout and Deep Ensembles.

## Baseline
1. `LSTM_CSP_model.py`: Model architecture file.
2. `LSTM_CSP_util.py`: Model parameter file.
* **Performance_Metrics**: Calculate the basic performance metrics of the model, including: NLL、RMSE、ADE and FDE.
* **UQ_Metrics**: Obtain the basic data for quantifying aleatoric and epistemic uncertainties in LSTM-CSP.
* **Safety_Metrics**: Calculate the safety metrics of the LSTM-CSP model, including: TET、TIT and CPI.


## Calculation
* Calculate the final uncertainty values and the Pearson correlation coefficient between epistemic uncertainty and ADS safety metrics.

## Highly uncertain samples
1. Overlap: First, run `1_Data_Production.py` and `2_Overlap_Calculation.py`, and then calculate the overlap of highly uncertain samples identified by MC Dropout and Deep Ensembles.
2. Ditribution: Calculate the distribution.
