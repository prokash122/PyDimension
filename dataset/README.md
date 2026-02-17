# Dataset Directory

This directory contains datasets and dimension matrices for different problems. Each problem should have its own subfolder.

## Structure

```
dataset/
├── translational_symmetry/      # Translational symmetry problem
│   ├── dataset_translational_symmetry.csv  # Translational symmetry dataset
│   └── dimension_matrix.csv              # Dimension matrix for translational symmetry problem
├── problem2/                   # Another problem (example)
│   ├── dataset_problem2.csv
│   └── dimension_matrix.csv
└── README.md                   # This file
```

## Adding a New Problem

1. Create a new subfolder: `dataset/your_problem_name/`
2. Place your dataset CSV file in that folder
3. Place your dimension matrix CSV file in that folder
4. Update your config file to point to the new paths:
   ```json
   {
     "DATA_PREPROCESSING": {
       "input_file": "dataset/your_problem_name/your_dataset.csv",
       "dimension_matrix_file": "dataset/your_problem_name/dimension_matrix.csv"
     }
   }
   ```

## Current Problems

### translational_symmetry
- **Dataset**: `dataset/translational_symmetry/dataset_translational_symmetry.csv`
- **Dimension Matrix**: `dataset/translational_symmetry/dimension_matrix.csv`
- **Config**: `pydimension/configs/config_translational_symmetry.json`
- **Description**: Translational symmetry example from the tutorial paper (update variable names/columns to match your dataset).

