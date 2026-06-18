# Base on LSTM-SE For Temperature Prediction
This project is based on the LSTM-SE model for temperature prediction.The model is appropriate for small sample forecast,it not need a large dataset like Transformer,and it not get influence from disorganized date.I was used BP to approve the same function,result is pretty bad,you can visit my [Base-On-BP-Neural-Network-Temperature-Forecast](https://github.com/KrisitVvv/Base-On-BP-Neural-Network-Temperature-Forecast) for more details.
## Device
In the model training process,I used a GPU with CUDA support.If you have NVIDIA GPU,you can install the CUDA environment to improve the training speed.But if you not have NVIDIA GPU,don't worry,in the code,I reserved the CPU version.

If you have conda environment,you can run the terminal command to install the required packages.
```bash
conda env create -f environment.yml
conda activate rt-ls
```
## Dataset
The dataset is from China Weather Data.You can run the terminal command to download the dataset.
```bash
python reptile.py --area chongqing \
-s ./reptile/output \
-b 2011 \ #begin year
-e 2025 \ #end year
--slow \  
```
I get the dataset for Chongqing,I process these data to final format in './reptile/output/pivot.'.You can skip the step to process to next step.
## Training and Evaluation
In this step,you can run the terminal command to train the model.
```bash
python LSTM-SE-Model.py -r ./reptile/output/pivot
```
After running the training script, the project structure will be as follows:
```
├── lstm_se_output/           # Model output directory
│   ├── images/               # Visualization images directory
│   │   ├── attention_channels_max.png      # Max temperature attention channel visualization
│   │   ├── attention_channels_min.png      # Min temperature attention channel visualization
│   │   ├── attention_weights_max.png       # Max temperature attention weights visualization
│   │   ├── attention_weights_min.png       # Min temperature attention weights visualization
│   │   ├── overall_max_temperature.png     # Overall max temperature trend chart
│   │   ├── overall_min_temperature.png     # Overall min temperature trend chart
│   │   ├── prediction_2026.png ~ prediction_2035.png  # Temperature prediction charts for 2026-2035
│   │   ├── test_comparison_max.png         # Max temperature test set comparison chart
│   │   ├── test_comparison_min.png         # Min temperature test set comparison chart
│   │   ├── training_metrics_max.png        # Max temperature training metrics chart
│   │   └── training_metrics_min.png        # Min temperature training metrics chart
│   ├── evaluation_metrics.json             # Model evaluation metrics JSON file
│   ├── lstm_se_max.pth                     # Max temperature prediction model weights
│   ├── lstm_se_min.pth                     # Min temperature prediction model weights
│   ├── prediction_max.xlsx                 # Max temperature prediction results Excel
│   └── prediction_min.xlsx                 # Min temperature prediction results Excel
├── reptile/                  # Crawler module directory
│   └── output/
│       └── pivot/
│           ├── MaxTemperature.xlsx         # Max temperature dataset
│           └── MinTemperature.xlsx         # Min temperature dataset
├── LSTM-SE-Model.py          # LSTM-SE model main program
├── README.md                 # Project documentation
├── environment.yml           # Conda environment configuration
├── msyh.ttc                  # Chinese font file
└── reptile.py                # Crawler script
```
After training the model,you can see the results images.You can intuitive monitoring the model performance.
<p align="center"><img src="https://github.com/user-attachments/assets/36dda1b4-543e-4af3-af1a-271dc867ba5b" width="800" height="297"></p>
<p align="center"><img src="https://github.com/user-attachments/assets/5ea0c11a-87b0-470f-96a0-e2b06b42c1fe" width="800" height="566"></p>
In addition to the training metrics,you can see the test set comparison charts.
<p align="center"><img src="https://github.com/user-attachments/assets/2d720145-f8fc-4bb8-a13c-9080cb3d8c03" width="800" height="281"></p>
And you can see the results images.
<p align="center"><img src="https://github.com/user-attachments/assets/8079a77c-5300-4ee5-a935-6f0b81f36bf9" width="800" height="297"></p>

## File Description

| File Path | Description |
| :--- | :--- |
| `lstm_se_output/images/` | Stores all visualization charts including attention mechanism visualization, prediction result charts, training metrics charts, etc. |
| `lstm_se_output/evaluation_metrics.json` | Contains model evaluation metrics on test set (MAE, MSE, RMSE, etc.) |
| `lstm_se_output/lstm_se_max.pth` | Trained max temperature prediction model weights |
| `lstm_se_output/lstm_se_min.pth` | Trained min temperature prediction model weights |
| `lstm_se_output/prediction_max.xlsx` | Max temperature prediction results with 10-year forecast data |
| `lstm_se_output/prediction_min.xlsx` | Min temperature prediction results with 10-year forecast data |
| `reptile/output/pivot/MaxTemperature.xlsx` | Crawled and processed max temperature dataset |
| `reptile/output/pivot/MinTemperature.xlsx` | Crawled and processed min temperature dataset |
| `LSTM-SE-Model.py` | Main program for LSTM-SE model training, evaluation, and prediction |
| `reptile.py` | Web crawler script for fetching weather data |
| `environment.yml` | Conda environment configuration file with all dependencies |
