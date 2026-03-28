import argparse
import statistics
import time


def _sync(cp):
    try:
        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def _timeit(cp, fn, repeats):
    times = []
    for _ in range(repeats):
        _sync(cp)
        start = time.perf_counter()
        fn()
        _sync(cp)
        times.append(time.perf_counter() - start)
    return times


def _summarize(name, times):
    med = statistics.median(times)
    mean = statistics.mean(times)
    return {
        "name": name,
        "median_seconds": med,
        "mean_seconds": mean,
    }


def main():
    parser = argparse.ArgumentParser(description="CatBoost native GPU input benchmark (CuPy).")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--cols", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    import cupy as cp
    from catboost import CatBoostRegressor

    cp.random.seed(0)
    x_gpu = cp.random.random((args.rows, args.cols), dtype=cp.float32)
    y_gpu = (x_gpu[:, 0] * 0.3 + x_gpu[:, 1] * -0.2 + 0.1).astype(cp.float32)

    params = dict(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices=args.device,
        random_seed=0,
        verbose=False,
        allow_writing_files=False,
    )

    def train_cpu_input():
        x_cpu = cp.asnumpy(x_gpu)
        y_cpu = cp.asnumpy(y_gpu)
        model = CatBoostRegressor(**params)
        model.fit(x_cpu, y_cpu)

    def train_gpu_input():
        model = CatBoostRegressor(**params)
        model.fit(x_gpu, y_gpu)

    warmup_params = dict(
        iterations=2,
        task_type="GPU",
        devices=args.device,
        verbose=False,
        allow_writing_files=False,
    )

    print(f"rows={args.rows} cols={args.cols} iters={args.iterations} depth={args.depth} device={args.device} repeats={args.repeats}")

    # Warm up CUDA/CatBoost for both CPU-input and GPU-input paths.
    _sync(cp)
    CatBoostRegressor(**warmup_params).fit(cp.asnumpy(x_gpu[:10_000]), cp.asnumpy(y_gpu[:10_000]))
    CatBoostRegressor(**warmup_params).fit(x_gpu[:10_000], y_gpu[:10_000])
    _sync(cp)

    cpu_times = _timeit(cp, train_cpu_input, args.repeats)
    gpu_times = _timeit(cp, train_gpu_input, args.repeats)

    cpu = _summarize("cpu_input", cpu_times)
    gpu = _summarize("gpu_input", gpu_times)

    for row in (cpu, gpu):
        print(
            f"{row['name']}: "
            f"median_seconds={row['median_seconds']:.3f} mean_seconds={row['mean_seconds']:.3f}"
        )

    speedup_median = cpu["median_seconds"] / gpu["median_seconds"] if gpu["median_seconds"] > 0 else float("inf")
    speedup_mean = cpu["mean_seconds"] / gpu["mean_seconds"] if gpu["mean_seconds"] > 0 else float("inf")
    print(f"speedup_median={speedup_median:.2f}x")
    print(f"speedup_mean={speedup_mean:.2f}x")


if __name__ == "__main__":
    main()
