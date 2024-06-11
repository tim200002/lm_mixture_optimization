import argparse
from pathlib import Path
import simdjson
import subprocess
import multiprocessing as mp
from tqdm import tqdm


def count_samples(shard_path):
    count = int(subprocess.check_output(f"tar tf {shard_path} | wc -l", shell=True))
    return count


def worker_fn(input_data):
    basename, data_dir = input_data
    shard_path = data_dir / basename
    return (
        basename,
        {
            "shard": basename.split(".")[0],
            "num_sequences": count_samples(shard_path),
        },
    )


def create_manifest(data_dir: str, num_workers: int = 2):
    data_dir = Path(data_dir)
    shards = sorted([x for x in data_dir.iterdir() if x.name.endswith(".tar")])
    input_data = [(shard.name, data_dir) for shard in shards]

    with mp.Pool(num_workers) as pool:
        data = []
        for worker_data in tqdm(pool.imap_unordered(worker_fn, input_data)):
            data.append(worker_data)

    data = sorted(data)
    data = [item[1] for item in data]
    manifest_filename = "manifest.jsonl"
    manifest_path = data_dir / manifest_filename
    with manifest_path.open("w") as fp:
        for item in data:
            simdjson.dump(item, fp)
            fp.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing a dataset in webdataset format.",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers.")
    args = parser.parse_args()

    create_manifest(args.data_dir, args.num_workers)
