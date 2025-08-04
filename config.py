config = {
    # Dataset Paths And Options
    'data_options': ['data/permittivity_Dataset'],  # modify to the actual path

    # Task type
    'task': 'regression',  # or 'classification'

    # Device settings
    'disable_cuda': False,

    # Data loading
    'workers': 0,
    'batch_size': 256,

    # Training parameters
    'epochs': 200,
    'start_epoch': 0,
    'lr': 0.001,
    'lr_milestones': [100],
    'momentum': 0.9,
    'weight_decay': 0,
    'print_freq': 20,
    'resume': '',  # Path to checkpoint for resuming training

    # Dataset splitting
    'train_ratio': 0.75,
    'train_size': None,
    'val_ratio': 0.25,
    'val_size': None,
    'test_ratio': 0,
    'test_size': None,

    # Optimizer
    'optim': 'SGD',  # or 'Adam'

    # Model architecture
    'atom_fea_len': 64,
    'h_fea_len': 128,
    'n_conv': 3,
    'n_h': 1,

    # ====== Prediction and Interpreter related parameters ======
    'modelpath': 'model_best.pth.tar',  # Path to trained model
    'cifpath': 'data/test_dataset',  # Data root directory
    'predict_batch_size': 128,  # Prediction batch size
    'predict_workers': 0,  # Number of workers for prediction data loading
    'predict_print_freq': 10,  # Prediction print frequency

    'explain_enabled': True,  # Whether to enable explanation
    'explain_global': True,  # Whether to run global feature importance analysis
    'explain_epochs': 200,  # Training epochs for GNNExplainer
    'explain_cif_name': 'Sr2SmSbO6_mp-972216_computed',  # CIF file name of material to explain (without .cif)
}
