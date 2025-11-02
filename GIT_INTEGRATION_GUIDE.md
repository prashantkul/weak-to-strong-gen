# Git Integration Guide

This guide explains how to use this codebase from a git repository in Jupyter notebooks and Google Colab.

## For Local Jupyter Notebooks

### Option 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/astra.git
cd astra

# Create conda environment
conda create -n astra python=3.10
conda activate astra

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# Launch Jupyter
jupyter notebook
```

Then in your notebook:

```python
import sys
from pathlib import Path

# Add project root to path (if not already)
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import modules
from src import Config, DatasetManager, ExperimentRunner
from notebook_experiments import run_baseline_sweep, run_disclaimer_sweep, run_cot_sweep

print("✓ Imports successful")
```

### Option 2: Install as Package

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/astra.git
cd astra

# Install as editable package
pip install -e .
```

Then in your notebook:

```python
# Direct import (no path manipulation needed)
from src import Config, DatasetManager, ExperimentRunner
```

## For Google Colab

### Option 1: Clone Repository in Colab

Add this cell at the beginning of your Colab notebook:

```python
# Clone repository
!git clone https://github.com/YOUR_USERNAME/astra.git
%cd astra

# Install dependencies
!pip install -q python-dotenv datasets pandas matplotlib
!pip install -q "safetytooling @ git+https://github.com/safety-research/safety-tooling.git@unpinned_requirements"

# Add to Python path
import sys
sys.path.insert(0, '/content/astra')

# Set environment variables
import os
os.environ['OPENROUTER_API_KEY'] = "your_key_here"
os.environ['OPENROUTER_API_KEY_BACKUP'] = "your_backup_key_here"

# Import and verify
from src import Config, DatasetManager, ExperimentRunner
from notebook_experiments import run_baseline_sweep, run_disclaimer_sweep, run_cot_sweep

print("✓ Setup complete")
```

### Option 2: Mount Google Drive + Clone

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone to Drive (persistent across sessions)
%cd /content/drive/MyDrive
!git clone https://github.com/YOUR_USERNAME/astra.git
%cd astra

# Install dependencies
!pip install -q -r requirements.txt

# Add to path
import sys
sys.path.insert(0, '/content/drive/MyDrive/astra')

# Use Colab Secrets for API keys (recommended)
from google.colab import userdata
import os
os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')
os.environ['OPENROUTER_API_KEY_BACKUP'] = userdata.get('OPENROUTER_API_KEY_BACKUP')

# Import modules
from src import Config, DatasetManager, ExperimentRunner
print("✓ Setup complete")
```

### Option 3: Pull Updates from Git

If you've already cloned and want to pull latest changes:

```python
# Navigate to repository
%cd /content/astra  # or /content/drive/MyDrive/astra

# Pull latest changes
!git pull origin main

# Restart runtime if needed (Runtime > Restart runtime)
```

## Repository Structure for Notebooks

When using the code from git, you have two options:

### Option A: Use Notebook Functions (Recommended)

Import the pre-built notebook-compatible functions:

```python
from notebook_experiments import (
    run_baseline_sweep,
    run_disclaimer_sweep,
    run_cot_sweep
)

# Run experiments
results_405b = await run_baseline_sweep(
    model_pair="8b_to_405b",
    k_values=[0, 2, 5, 10],
    save_results=True
)
```

### Option B: Use Core Library Directly

Import the core classes and build your own experiments:

```python
from src import Config, DatasetManager, ExperimentRunner, ModelEvaluator

# Setup
config = Config.from_env()
config.setup_environment()

# Load data
dm = DatasetManager()
test_data, few_shot_pool, split = dm.load_split()

# Run custom experiment
runner = ExperimentRunner(config)
results = await runner.run_few_shot_experiment(...)
```

## Environment Variables

### Local Development

Create a `.env` file in the repository root:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_API_KEY_BACKUP=sk-or-v1-...
```

The `Config.from_env()` will automatically load these.

### Google Colab

**Option 1: Hardcode in Notebook** (not recommended for sharing)

```python
import os
os.environ['OPENROUTER_API_KEY'] = "sk-or-v1-..."
os.environ['OPENROUTER_API_KEY_BACKUP'] = "sk-or-v1-..."
```

**Option 2: Use Colab Secrets** (recommended)

1. Click the key icon in left sidebar
2. Add secrets:
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_API_KEY_BACKUP`
3. In notebook:

```python
from google.colab import userdata
import os
os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')
os.environ['OPENROUTER_API_KEY_BACKUP'] = userdata.get('OPENROUTER_API_KEY_BACKUP')
```

## Complete Colab Setup Example

Here's a complete first cell for a Colab notebook:

```python
# === SETUP CELL - RUN FIRST ===

# Clone repository
!git clone https://github.com/YOUR_USERNAME/astra.git
%cd astra

# Install dependencies
!pip install -q python-dotenv datasets pandas matplotlib pydantic asyncio-throttle tqdm
!pip install -q "safetytooling @ git+https://github.com/safety-research/safety-tooling.git@unpinned_requirements"

# Add to Python path
import sys
sys.path.insert(0, '/content/astra')

# Set API keys (using Colab Secrets)
from google.colab import userdata
import os
os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')
os.environ['OPENROUTER_API_KEY_BACKUP'] = userdata.get('OPENROUTER_API_KEY_BACKUP')

# Verify imports
from src import Config, DatasetManager, ExperimentRunner
from notebook_experiments import run_baseline_sweep, run_disclaimer_sweep, run_cot_sweep

# Initialize config
config = Config.from_env()
config.setup_environment()

print("=" * 70)
print("✓ Repository cloned")
print("✓ Dependencies installed")
print("✓ Imports successful")
print("✓ Configuration loaded")
print(f"✓ Weak model: {config.weak_model}")
print(f"✓ Strong model: {config.strong_model}")
print("=" * 70)
print("\nReady to run experiments!")
```

## Updating Code from Git

### In Colab

```python
# Pull latest changes
%cd /content/astra
!git pull origin main

# Restart runtime to reload modules
# (Runtime > Restart runtime in Colab menu)
```

### In Local Jupyter

```bash
# In terminal
cd /path/to/astra
git pull origin main

# In Jupyter, restart kernel
# (Kernel > Restart in Jupyter menu)
```

## Common Issues and Solutions

### Issue: Import errors after git pull

**Solution**: Restart the kernel/runtime to reload modules

```python
# Colab: Runtime > Restart runtime
# Jupyter: Kernel > Restart
```

### Issue: Module not found

**Solution**: Verify path is added correctly

```python
import sys
print(sys.path)  # Should include repository path

# If not, add it:
sys.path.insert(0, '/content/astra')  # Colab
# or
sys.path.insert(0, '/path/to/astra')  # Local
```

### Issue: Environment variables not loaded

**Solution**: Check environment variable is set

```python
import os
print(os.getenv('OPENROUTER_API_KEY'))  # Should print your key

# If None, set it:
os.environ['OPENROUTER_API_KEY'] = "your_key"
```

### Issue: Dependencies missing

**Solution**: Install requirements

```bash
pip install -r requirements.txt
```

## Git Workflow for Development

### Making Changes

```bash
# Create a new branch
git checkout -b feature/my-changes

# Make changes to code
# ...

# Commit changes
git add .
git commit -m "Description of changes"

# Push to GitHub
git push origin feature/my-changes

# Create pull request on GitHub
```

### Pulling Changes in Notebook

```python
# Pull latest from main branch
%cd /content/astra
!git checkout main
!git pull origin main

# Restart runtime to reload
```

## Repository Structure Reference

```
astra/
├── src/                          # Core library (import this)
├── notebook_experiments.py       # Notebook-compatible functions
├── test_notebook_functions.py    # Quick test script
├── run_*.py                      # Standalone experiment scripts
├── generate_*.py                 # Label generation scripts
├── create_final_comparison.py    # Visualization script
├── requirements.txt              # Dependencies
├── .env.example                  # Example environment file
└── Complete_Weak_to_Strong_Experiments.ipynb  # Full notebook
```

## Best Practices

1. **Always pull before starting work**: `git pull origin main`
2. **Use Colab Secrets for API keys**: Never hardcode keys in shared notebooks
3. **Restart runtime after git pull**: Ensures modules are reloaded
4. **Use notebook_experiments.py**: Pre-built functions work better in Jupyter
5. **Check environment setup**: Run verification cell before experiments
6. **Save results**: All experiments auto-save to `results/` directory

## Quick Test

To verify everything is working:

```python
# Run quick test
from test_notebook_functions import test_notebook_setup
await test_notebook_setup()
```

This will verify:
- Imports work
- Config loads
- Dataset loads
- API calls work

If all tests pass, you're ready to run the full experiments!
