# Project Description
This project implements an anomaly detection system using autoencoders. The system is designed to learn patterns from normal data and detect deviations as anomalies. It includes:

- Data loading and preprocessing pipelines.
- A customizable autoencoder model for anomaly detection.
- Training utilities with checkpointing and early stopping.

## Setup and Testing

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Sample Data**:
   Run the following command to automatically download and prepare sample data:
   ```bash
   python main.py --download-data
   ```

3. **Run the Main Script**:
   ```bash
   python main.py
   ```

4. **Verify Outputs**:
   - Model checkpoints should appear in `models/checkpoints/`.
   - Final saved model will be located at `models/saved_model.h5`.