from waitress import serve
from deploy import app
import multiprocessing

if __name__ == "__main__":
    # get cpu count
    num_cpus = multiprocessing.cpu_count()

    threads_per_worker = max(1, num_cpus-1)

    print("Threads:", threads_per_worker)
    print('Server Started')
    serve(
        app,
        host='0.0.0.0',
        port=8080,
        threads=threads_per_worker
    )