# CGEN-model
project/
├── data/
│   ├── cif/               # CIF file directory
│   │   ├── id0.cif
│   │   ├── id1.cif
│   │   ├── ...
│   ├── atom_init.json
│   ├── train.csv
│   ├── val.csv
│   └── test.csv            # or id_prop.csv
├── model/
│   ├── cgcnn.py            # Model definition
│   ├── GNNExplainer.py     # GNN Explainer
│   └── data.py             # data processing
├── config.py               # Configurations for training/predicting/explaining
├── train.py                # Training script
├── model_best.pth.tar      # Best saved model checkpoint
├── prediction_module.py    # Predict scripts using trained models
└── explanation_module.py   # Explainable scripts
