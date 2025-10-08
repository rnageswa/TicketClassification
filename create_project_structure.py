from pathlib import Path

# Define the root directory name for the project
PROJECT_ROOT = Path("TicketClassification")

# Define the directories to create
directories = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "processed" / "triage_data",
    PROJECT_ROOT / "data" / "processed" / "solution_data",
    PROJECT_ROOT / "notebooks",
    PROJECT_ROOT / "src" / "data_processing",
    PROJECT_ROOT / "src" / "models",
    PROJECT_ROOT / "src" / "training",
    PROJECT_ROOT / "src" / "utils",
    PROJECT_ROOT / "config",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "artifacts" / "triage_model",
    PROJECT_ROOT / "artifacts" / "solution_dssm",
    PROJECT_ROOT / "artifacts" / "predictive_lstm",
]

# Define the files to create, including their initial content
files = [
    # Data files
    (PROJECT_ROOT / "data" / "raw" / "support_tickets.csv", ""),
    (PROJECT_ROOT / "data" / "processed" / "tickets_clean.csv", ""),
    (PROJECT_ROOT / "data" / "processed" / "triage_data" / "train.csv", ""),
    (PROJECT_ROOT / "data" / "processed" / "triage_data" / "test.csv", ""),
    (PROJECT_ROOT / "data" / "processed" / "solution_data" / "dssm_train_queries.txt", ""),
    (PROJECT_ROOT / "data" / "processed" / "solution_data" / "dssm_train_documents.txt", ""),

    # Notebooks
    (PROJECT_ROOT / "notebooks" / "1_data_exploration.ipynb", "# Jupyter notebook for data exploration"),
    (PROJECT_ROOT / "notebooks" / "2_triage_model_training.ipynb", "# Jupyter notebook for triage model training"),
    (PROJECT_ROOT / "notebooks" / "3_solution_model_training.ipynb", "# Jupyter notebook for solution model training"),
    (PROJECT_ROOT / "notebooks" / "4_predictive_model_training.ipynb", "# Jupyter notebook for predictive model training"),
    
    # Source code files
    (PROJECT_ROOT / "src" / "__init__.py", ""),
    (PROJECT_ROOT / "src" / "data_processing" / "__init__.py", ""),
    (PROJECT_ROOT / "src" / "data_processing" / "preprocess.py", "# Script for data preprocessing"),
    (PROJECT_ROOT / "src" / "models" / "__init__.py", ""),
    (PROJECT_ROOT / "src" / "models" / "triage_model.py", "# Script for the triage model"),
    (PROJECT_ROOT / "src" / "models" / "solution_dssm.py", "# Script for the solution DSSM model"),
    (PROJECT_ROOT / "src" / "models" / "predictive_lstm.py", "# Script for the predictive LSTM model"),
    (PROJECT_ROOT / "src" / "training" / "__init__.py", ""),
    (PROJECT_ROOT / "src" / "training" / "train_triage.py", "# Script for training the triage model"),
    (PROJECT_ROOT / "src" / "utils" / "__init__.py", ""),
    (PROJECT_ROOT / "src" / "utils" / "metrics.py", "# Script for evaluation metrics"),
    
    # Configuration files
    (PROJECT_ROOT / "config" / "model_configs.json", "{}\n"),
    (PROJECT_ROOT / "config" / "training_params.json", "{}\n"),
    (PROJECT_ROOT / "config" / "feature_engineering.yaml", "---\n"),
    
    # Scripts
    (PROJECT_ROOT / "scripts" / "run_training.sh", "#!/bin/bash\n# Script to run the model training pipeline"),
    
    # Artifacts (empty placeholders)
    (PROJECT_ROOT / "artifacts" / "triage_model" / "best_model.h5", ""),
    (PROJECT_ROOT / "artifacts" / "triage_model" / "metadata.json", "{}\n"),
    
    # Root files
    (PROJECT_ROOT / ".gitignore", """# Python\n__pycache__/\n*.pyc\n\n# IDE files\n.vscode/\n.idea/\n\n# Data\ndata/processed/\n\n# Artifacts\nartifacts/\n"""),
    (PROJECT_ROOT / "requirements.txt", ""),
    (PROJECT_ROOT / "README.md", "# Ticket Intelligence Project\n\n## Overview\nThis project uses deep learning to improve technical support operations.\n"),
]

def create_project_structure():
    """
    Creates the project's folder and file structure.
    """
    print("Creating project directories...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("Creating project files...")
    for file_path, content in files:
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    print(f"Project structure for '{PROJECT_ROOT}' has been successfully created.")
    
if __name__ == "__main__":
    create_project_structure()
