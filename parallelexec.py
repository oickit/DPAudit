import multiprocessing
import subprocess

"""
parallel execution
"""

def run_script():
    subprocess.run(["python", "auditlaplacewithlr.py"])

if __name__ == "__main__":
    processes = []
    for i in range(4):
        p = multiprocessing.Process(target=run_script)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
