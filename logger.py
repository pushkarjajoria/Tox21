import os
import sys
from datetime import datetime

# Define the log directory and file globally (so it remains consistent across function calls)
log_dir = "logs"
log_file = None  # To be defined when print() is called for the first time
# Backup the original print function to use later
built_in_print = print


# Custom print function that overrides the built-in print
def custom_print(*args, **kwargs):
    global log_file

    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create log file if it's not yet created
    if log_file is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = os.path.join(log_dir, f"run_log_{timestamp}.txt")

    # Call the original print function to print to the console
    built_in_print(*args, **kwargs)

    # Prepare the output to be written to the log file
    output = ' '.join(map(str, args))  # Join all arguments into a string
    with open(log_file, 'a') as f:
        f.write(output + '\n')

