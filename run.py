import os
import subprocess

if __name__ == "__main__":
    print("Running Thalamus Benchmark Environment. Use the provided notebook link to access the environment in your browser.")
    code_path = os.path.abspath("./code")
    dst_path = "/home/code"
    subprocess.run(["sudo", "docker", "run", "-p", "8888:8888", "-v", f"{code_path}:{dst_path}", "--shm-size=1024m", "--gpus", "all", "-it", "thalamus_env"])
    