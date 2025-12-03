import os
import argparse
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import random
import snakes2
from tqdm.auto import tqdm
import contextlib
import io


def _worker_run(worker_id: int, base_kwargs: dict, batch_id: int, n_images: int, seed: int, suppress_output: bool):
    # Per-process seeding for deterministic variability across workers
    np.random.seed(seed)
    random.seed(seed)
    a = SimpleNamespace(**base_kwargs)
    a.batch_id = int(batch_id)
    a.n_images = int(n_images)
    if suppress_output:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            snakes2.from_wrapper(a)
    else:
        snakes2.from_wrapper(a)
    return {"worker": worker_id, "batch_id": int(batch_id), "count": int(n_images)}


def ensure_integer_thickness(thickness: float, aa: int) -> tuple[float, int]:
    prod = thickness * aa
    if abs(round(prod) - prod) <= 1e-6:
        return thickness, aa
    # try a few AA values to make product integral
    for aa_try in [2, 3, 4, 6, 8]:
        prod_try = thickness * aa_try
        if abs(round(prod_try) - prod_try) <= 1e-6:
            return thickness, aa_try
    # fallback: round thickness
    return float(round(thickness)), max(1, aa)


def main():
    p = argparse.ArgumentParser(description="Generate 128x128 tight dashed snakes using snakes2")
    p.add_argument('--out', type=str, default='./Data/Snakes128_tight', help='Output root folder')
    p.add_argument('--n', type=int, default=1000, help='Number of images to generate')
    p.add_argument('--batch', type=int, default=0, help='Batch id (subfolder under imgs/)')
    p.add_argument('--size', type=int, default=128, help='Image size (square)')
    p.add_argument('--aa', type=int, default=2, help='Antialias scaling (ensures thickness*aa is integer)')
    p.add_argument('--thickness', type=float, default=1, help='Dash thickness (float allowed, e.g. 1.5)')
    p.add_argument('--paddle-length', type=int, default=6, help='Dash length in pixels')
    p.add_argument('--margin', type=int, default=2, help='Gap between dashes (lower=tighter)')
    p.add_argument('--continuity', type=float, default=1.2, help='Smoothness; 1.2–1.6 gives long smooth arcs')
    p.add_argument('--contour-length', type=int, default=14, help='Number of segments in main snakes')
    p.add_argument('--distractor-length', type=int, default=4, help='Length of distractor snakes')
    p.add_argument('--num-distractors', type=int, default=8, help='How many distractor snakes')
    p.add_argument('--seed-distance', type=int, default=14, help='Min distance between two target seeds')
    p.add_argument('--padding', type=int, default=4, help='Border padding (keep-out)')
    p.add_argument('--no-single-paddles', action='store_true', help='Disable extra single paddles (recommended)')
    p.add_argument('--marker-radius', type=int, default=3, help='Endpoint marker radius')
    p.add_argument('--workers', type=int, default=None, help='Number of parallel processes (defaults to CPU count)')
    p.add_argument('--seed', type=int, default=42, help='Base RNG seed for reproducibility across workers')
    p.add_argument('--verbose', action='store_true', help='Print generator logs (default: suppressed)')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    thickness, aa = ensure_integer_thickness(args.thickness, args.aa)

    use_single_paddles = not args.no_single_paddles

    # Build args recognized by snakes2.from_wrapper
    a = SimpleNamespace(
        # IO
        contour_path=args.out,
        batch_id=int(args.batch),
        n_images=int(args.n),
        save_images=True,
        save_metadata=True,
        pause_display=False,
        segmentation_task=False,
        segmentation_task_double_circle=False,

        # Canvas / AA
        window_size=[int(args.size), int(args.size)],
        padding=int(args.padding),
        antialias_scale=int(aa),

        # Markers
        LABEL=1,
        marker_radius=int(args.marker_radius),

        # Geometry / difficulty (tight and smooth)
        continuity=float(args.continuity),
        contour_length=int(args.contour_length),
        distractor_length=int(args.distractor_length),
        num_distractor_snakes=int(args.num_distractors),
        snake_contrast_list=[1.0],

        # Dash parameters
        paddle_length=int(args.paddle_length),
        paddle_thickness=float(thickness),
        paddle_margin_list=[int(args.margin)],
        paddle_contrast_list=[1.0],
        use_single_paddles=use_single_paddles,

        # Placement / retries
        seed_distance=int(args.seed_distance),
        max_target_contour_retrial=4,
        max_distractor_contour_retrial=4,
        max_paddle_retrial=2,
    )

    # Parallel or single-process execution with progress bar
    cpu_count = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else cpu_count
    workers = max(1, min(int(workers), int(args.n), cpu_count))
    base_kwargs = a.__dict__.copy()
    total = int(args.n)

    if workers == 1:
        # Serial generation: chunk into batches and report progress
        chunk_size = max(1, min(200, total))
        remaining = total
        counts = []
        while remaining > 0:
            take = min(chunk_size, remaining)
            counts.append(take)
            remaining -= take

        results = []
        with tqdm(total=total, desc="Generating", unit="img") as pbar:
            for i, c in enumerate(counts):
                batch_id = int(args.batch) + i
                seed = int(args.seed) + i
                res = _worker_run(i, base_kwargs, batch_id, c, seed, suppress_output=(not args.verbose))
                results.append(res)
                pbar.update(c)

        made = sum(r.get("count", 0) for r in results)
        batches = ", ".join(str(r.get("batch_id")) for r in results)
        print(f"Done -> {args.out} | images: {made}/{total} | batches: {batches}")
        return

    # Parallel execution with progress bar
    base = total // workers
    rem = total % workers
    counts = [base + (1 if i < rem else 0) for i in range(workers)]
    counts = [c for c in counts if c > 0]

    print(f"Spawning {len(counts)} workers on {cpu_count} CPUs...")
    results = []
    with ProcessPoolExecutor(max_workers=len(counts)) as ex:
        futs = []
        for i, c in enumerate(counts):
            batch_id = int(args.batch) + i
            seed = int(args.seed) + i
            futs.append(ex.submit(_worker_run, i, base_kwargs, batch_id, c, seed, not args.verbose))
        with tqdm(total=total, desc="Generating", unit="img") as pbar:
            for f in as_completed(futs):
                try:
                    r = f.result()
                    results.append(r)
                    pbar.update(r.get("count", 0))
                except Exception as e:
                    print(f"Worker failed: {e}")

    made = sum(r.get("count", 0) for r in results)
    batches = ", ".join(str(r.get("batch_id")) for r in sorted(results, key=lambda r: r["worker"]))
    print(f"Done -> {args.out} | images: {made}/{total} | batches: {batches}")


if __name__ == '__main__':
    main()
