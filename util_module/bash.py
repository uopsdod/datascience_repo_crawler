import subprocess

"""
    if shell if True, the specified command will be executed through shell
    if check is True, and the process exists with a non-zero exist code, a CalledProcessError exception will be raised 
    if capture_output is True, stdout and stderr will be captured 
    
    if return_code is non-zero, raise a CaledProcessError 
"""

def run_command(command, verbose=True):
    # TODO: check return for robustness purposes
    subprocess.run(command, shell=True, check=True, capture_output=not verbose)


