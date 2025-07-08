
## 🗂️ Repository Structure
```bash
.
├── config                    # Configuration files for your project
│   └── config.yaml           # Example configuration file
├── data                      # Folder to store raw and processed data
│   ├── database              # Databases or local data storage
│   ├── processed             # Preprocessed/cleaned data
│   └── raw                   # Raw data inputs
├── iac                       # Infrastructure as Code (IaC) scripts for cloud deployment
├── notebooks                 # Jupyter notebooks for exploratory data analysis, experiments
│   └── 00_example.ipynb      # Example notebook
├── results                   # Folder to store the results of experiments and models
├── src                       # Source code of your project
│   ├── models                # Machine learning model scripts
│   ├── pipelines             # ML pipelines for preprocessing and modeling
│   ├── schemas               # Data schemas and validation logic
│   ├── utils                 # Utility functions
├── tests                     # Unit and integration tests
│   └── test_example.py       # Example test file using pytest
├── train.py                  # train.py
├── .env                      # Environment variables file
├── .gitignore                # Standard .gitignore file
├── .pre-commit-config.yaml   # Configuration for pre-commit hooks
├── .python-version           # Python version for the project
├── docker-compose.yaml       # Docker Compose setup for MLflow, OpenSearch and related services
├── pyproject.toml            # Configuration for formatting, linting, type-checking, and testing
├── README.md                 # Documentation for the project
└── uv.lock                   # Lock file for uv package manager
