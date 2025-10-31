## Repo Structure

This is the current envisioned repo structure we want to maintain. 
Where needed we may restructure to make it more relevant to the repository examples.

```

cv-playground/
├── notebooks/
│   ├── {cv-playground use case example folder #0}
│   └── NuInsSeg/
│       └── README.md        ## CV-Playground Use Case Folder README
│       └── InstanceSegmentation_sgc/
│           ├── CellTypes_InstanceSeg_TransferLearn_serverlessA10_v0.3.3.py
│           ├── data.yaml
│           ├── datasets/  # YOLO-formatted image datasets (train/val/test)
│           ├── imgs/
│           └── utils/     # Modular utility functions
│               ├── mlflow_callbacks.py     # MLflow integration & checkpointing
│               ├── inference_utils.py      # Model inference utilities
│               ├── visualization_utils.py  # Results visualization
│               ├── yolo_utils.py           # YOLO path & environment setup
│               ├── summary_utils.py        # Reporting utilities
│               ├── cache_utils.py          # CUDA/memory management
│               └── resume_callbacks.py     # Training resume utilities
│           ├── {CellTypes_InstanceSeg_TransferLearn_serverless###_multinode}_.py    #forthcoming
│       └── InstanceSegmentation_classic/                                            #forthcoming
│           ├── {CustomData_to_YOLO_processing}/                                     #forthcoming
│               ├── {01_...}.py     
│               ├── {02_...}.py     
│               ├── {03_...}.py
│           ├── ...     
│           ├── {CellTypes_InstanceSeg_TransferLearn_classicGPU}_.py                 #forthcoming
│           ├── {CellTypes_InstanceSeg_TransferLearn_classicGPU_multinode}_.py       #forthcoming
│           ├── ...
│   ├── {cv-playground application example folder#1}
│   ├── {cv-playground application example folder#2}
│   ├── ...
└── README.md        ## Global README
└── ... 
  
```
