# Specifies the project name under which the job is run. CDRHID0024 is the project code.
#$ -P CDRHID0024

#  Runs the job in the current working directory.
# Can be changed so that job runs in a specific directory
# -cwd /path/to/your/working/directory
#$ -cwd

# Specifies the shell to interpret the script. In this case, it uses /bin/sh.
# Can be changed to -S /bin/bash to use bash instead of sh
#$ -S /bin/sh

# Requests resources for the job.
# h_vmem=2G requests 2 GB of virtual memory.
# h_rt=048:00:00 requests a runtime limit of 48 hours
#$ -l h_vmem=2G,h_rt=048:00:00

# Merges the standard error stream with the standard output stream, meaning both will be written to the same output file.
#$ -j y

# (Commented out) Would enable email notifications at the end (e) and when the job is aborted (a).
# can be changed to -M mukul.sherekar@fda.hhs.gov
#$ -m ea

# Sets the name of the job to rngstream.
#$ -N rngstream

# how to change sysout because code has destination folders inbuilt in it
# Specifies the file where the standard output (and standard error, since they are merged)
# will be written. The output will be written to a file named sysout.
#$ -o sysout

#  Requests a parallel environment with 1 thread. This is useful for multi-threaded applications,
#  but here it requests a single-thread environment. Can ask for more threads.
#$ -pe thread 1

# Submits a job array with tasks numbered from 1 to 3. Each task will run as a separate job with its own task ID.
#$ -t  1-3

# Prints a message indicating the task ID, job name, job ID, and the hostname of the machine running the job.
# SGE_TASK_ID, JOB_NAME, JOB_ID, and HOSTNAME are environment variables provided by the job scheduler.
echo "Running task $SGE_TASK_ID of job $JOB_NAME (ID of $JOB_ID) on $HOSTNAME"

# what is meaning of this line? is it activating an environment of python 3.10?
# Sources the script to set up the environment for running the Python application. This likely sets up the
# necessary paths and environment variables for the Python 3.10 environment on CentOS 7.

# how can this change to  virtual environment?
source /projects/mikem/applications/centos7/python3.10/set_run_env.sh

# Runs the Python script rngstream.py and measures the time it takes to execute, using the time command.
# can be changed to to run a differenct script and/or pass argument
# time python3 preprocess_and_train_device_models.py arg1 arg2
time python3 preprocess_and_train_device_models.py
