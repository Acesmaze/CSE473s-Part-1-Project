# CSE473s — Build Your Own Neural Network Library

## Structure
```
├── lib/                  # Custom NumPy neural network library
│   ├── __init__.py
│   ├── layers.py         # Layer base class + Dense
│   ├── activations.py    # ReLU, Sigmoid, Tanh, Softmax, Dropout
│   ├── losses.py         # MSE, Binary Cross-Entropy
│   ├── optimizer.py      # SGD with Momentum
│   └── network.py        # Sequential model
├── notebooks/
│   └── part1_xor.ipynb   # Part 1: Gradient checking + XOR + TF comparison
├── report/               # Generated plots
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
jupyter notebook notebooks/part1_xor.ipynb
```

## Part 1 Results
- Gradient check max relative error: **1.55e-9** (threshold: 1e-5) ✅
- XOR accuracy: **4/4 (100%)** ✅
