# Temperature-Forecast-RNN

Autoregressive time-series forecasting using a custom sliding-window dataset and RNN in PyTorch.

---

## Overview

This project explores time-series forecasting using Recurrent Neural Networks (RNNs) implemented in PyTorch. The objective was to understand how sequence models process temporal data and how autoregressive prediction can be implemented using sliding windows over historical observations.

Rather than focusing purely on model performance, this project emphasizes understanding sequence modeling mechanics — including dataset construction, tensor shapes, and how RNNs unfold across time.

---

## Project Overview

The model learns to predict future temperature values from past observations using daily minimum temperature data from Melbourne.

The forecasting approach follows an autoregressive setup:
```
Past observations → Model → Next values
```

Instead of predicting a single step ahead, the model predicts a shifted version of the input sequence, allowing the network to learn how temperature evolves over time.

---

## Dataset

**Dataset:** Daily Minimum Temperatures in Melbourne  
**Source:** https://github.com/jbrownlee/Datasets  

The dataset contains daily minimum temperatures recorded from 1981 to 1990.

### Example

| Date       | Temp |
|------------|------|
| 1981-01-01 | 20.7 |
| 1981-01-02 | 17.9 |
| 1981-01-03 | 18.8 |

---

## Data Preparation

To train the model on sequences, the dataset is converted into sliding windows using a custom PyTorch `Dataset`.

**Window size used:**

`window_size = 50`

Each training sample is structured as:
```
Input : [t1, t2, t3 ... t50]
Target : [t2, t3, t4 ... t51]
```

This setup allows the model to learn temporal transitions between consecutive values.

---

## Custom TimeSeriesDataset

```python
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, index):
        x = self.data[index:index+self.window_size]
        y = self.data[index+1:index+self.window_size+1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
```
## Model Architecture

The forecasting model is a multi-layer vanilla RNN.

### Architecture
```
Input Layer
↓
3-layer RNN (hidden size = 20)
↓
Linear Layer
↓
Predicted temperature sequence
```

### Implementation

```python
class RNN(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=20, num_layers=3):
        super().__init__()
        self.RNN = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.RNN(x)
        out = self.Linear(out)
        return out.squeeze(-1)
```
# RNN Temperature Time-Series Forecasting

This project explores the implementation of a **Recurrent Neural Network (RNN)** to predict temperature fluctuations. The primary objective was to move beyond high-level abstractions and understand the underlying mechanics of **RNN Unfolding**, hidden state propagation, and tensor transformations.

---

## Understanding RNN Unfolding

A central pillar of this project is the concept of **RNN Unfolding** (or unrolling). While diagrams often depict multiple recurrent units side-by-side, the physical architecture consists of a **single recurrent unit** whose parameters ($W$, $U$, and $b$) are shared across every timestep.



**The Core Recurrence Relation:**
$$h_t = \sigma(W x_t + U h_{t-1} + b)$$

The RNN processes a sequence by looping the same unit across timesteps, using the **hidden state** ($h$) as a "memory" that carries information forward from the past to the present.

---

## Tensor Shape Reasoning

A major technical focus was mapping how data dimensions evolve through the pipeline. Precision in shapes is the difference between a working model and a `RuntimeError`.

### 1. Input Tensor
The model expects a 3D tensor representing the temporal window:
` (batch_size, sequence_length, input_size) `

* **Example Case:** `(64, 50, 1)`
* **batch_size (64):** Number of independent sequences processed in one parallel pass.
* **sequence_length (50):** The look-back window (e.g., the last 50 hours of data).
* **input_size (1):** The number of features per timestep (Temperature).

### 2. Layer Transitions
The following table tracks the transformation of data as it moves through the network:

| Layer | Input Shape | Output Shape | Logic |
| :--- | :--- | :--- | :--- |
| **RNN Layer** | `(64, 50, 1)` | `(64, 50, hidden_size)` | Features extracted for every step in the sequence. |
| **Linear (FC)** | `(64, hidden_size)` | `(64, 1)` | Mapping the final hidden state to a single prediction. |

---

## Training Setup

The training environment was configured to optimize for convergence stability over 120 iterations.

* **Loss Function:** `Mean Squared Error (MSE)` — Ideal for regression tasks.
* **Optimizer:** `Stochastic Gradient Descent (SGD)`
* **Learning Rate:** `0.01`
* **Epochs:** `120`
* **Checkpointing:** The pipeline includes automated logic to save model weights, allowing for interrupted sessions to resume.

---

## Results & Visualization

The model performance is validated by comparing the predicted temperature against the ground truth on a held-out test set.



* **Temporal Trends:** The model effectively captures daily seasonality.
* **Error Analysis:** Qualitative plots show the RNN's ability to "smooth" noise while maintaining the primary trend.

---

## Implementation Snippet

```python
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        # The core recurrent unit
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initializing hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
```
