# Model Evaluation Plan

## Models to Compare
1. Neural Network (existing implementation)
2. Random Forest Regressor
3. K-Nearest Neighbors
4. XGBoost Regressor

## Data Preprocessing
- Use consistent preprocessing across all models:
  * Replace -200 RSSI values with -150 (out-of-range indicator)
  * Apply MinMaxScaler to RSSI values
  * Encode location labels consistently
  * Use same train/test split for fair comparison

## Evaluation Process
1. Load and preprocess data using existing pipeline
2. Train neural network model with timing measurements:
   - Add timing decorators or context managers to measure:
     * Training time (including autoencoder pretraining)
     * Inference time per sample
     * Total execution time
3. Train traditional models using model_comparison.py
4. Evaluate all models using consistent metrics:
   - Mean Distance Error
   - Median Distance Error
   - 90th Percentile Error
   - Training Time (including all preprocessing steps)
   - Inference Time (per sample)
   - Total Execution Time

## Implementation Steps
1. Create timing utility functions:
   def measure_time(func):
       def wrapper(*args, **kwargs):
           start_time = time.time()
           result = func(*args, **kwargs)
           end_time = time.time()
           return result, end_time - start_time
       return wrapper

2. Modify neural network training code:
   - Add timing decorator to train_model function
   - Add timing measurements for prediction
   - Store timing results consistently

3. Create main comparison script that:
   - Loads and preprocesses data
   - Trains all models with timing measurements
   - Evaluates using consistent metrics
   - Generates comparison tables and visualizations

4. Implement visualization functions:
   - Error distribution plots
   - Training time comparison bar charts
   - Inference time comparison bar charts

## Expected Outputs
1. Comparison table with all metrics for each model
2. Error distribution plots for each model
3. Training and inference time comparisons
4. Detailed analysis of strengths/weaknesses of each approach

## Timing Implementation Details
1. Neural Network Timing:
   - Measure autoencoder training time
   - Measure location predictor training time (both frozen and unfrozen)
   - Measure inference time using batch size=1 for per-sample timing
   - Record total execution time

2. Traditional Models Timing:
   - Measure preprocessing time
   - Measure training time
   - Measure inference time (per sample)
   - Record total execution time
