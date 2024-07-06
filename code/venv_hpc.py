# Navigate to your project directory on the HPC system and create a virtual environment
python3 -m venv myenv

# Activate the Virtual Environment
source myenv/bin/activate

# Install Dependencies:
pip install -r requirements.txt

# Update your HPC job script
# Load necessary modules
source /projects/mikem/applications/centos7/python3.10/set_run_env.sh
# is python 3.9 available? check with mike.

# Activate the virtual environment
source /path/to/your/project/myenv/bin/activate

# Run your script
python /path/to/your/project/main.py

# Deactivate virtual environment after finishing
deactivate
