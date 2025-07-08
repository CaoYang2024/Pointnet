
## 🗂️ Repository Structure
```bash
.
├── config                    # Configuration files for your project
│   └── config.yaml           # Example configuration file
│                             # Multiple configs allow Hydra to compose configurations for running various experiments
├── data                      # Folder to store raw and processed data
│   ├── database              # Databases or local data storage (tracked by DVC)
│   ├── processed             # Preprocessed/cleaned data (tracked by DVC)
│   └── raw                   # Raw data inputs (tracked by DVC)
├── iac                       # Infrastructure as Code (IaC) scripts for cloud deployment
├── notebooks                 # Jupyter notebooks for exploratory data analysis and experiments
│   └── 00_example.ipynb      # Example notebook
├── results                   # Folder to store results of experiments and trained models
│   └── ckpt.pth              # checkpoint documents
├── src                       # Source code of your project
│   ├── models                # Machine learning model definitions
│   ├── pipelines             # ML pipelines for data preprocessing and training
│   ├── schemas               # Data schemas and validation logic
│   ├── utils                 # Utility functions
├── tests                     # Unit and integration tests
│   └── test_example.py       # Example test file using pytest
├── train.py                  # Entry point for training (integrated with wandb logging)
├── .env                      # Environment variables file
├── wandb                     # wandb log
├── .gitignore                # Git ignore rules
├── .pre-commit-config.yaml   # Configuration for pre-commit hooks
├── docker-compose.yaml       # Docker Compose setup for MLflow, OpenSearch, and related services
├── pyproject.toml            # Project configuration for formatting, linting, type-checking, and testing
├── README.md                 # Documentation and usage instructions for the project
└── uv.lock                   # Lock file for the uv package manager

