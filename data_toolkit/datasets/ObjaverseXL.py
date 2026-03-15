import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import objaverse.xl as oxl


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata


def download(metadata, *, download_root, root=None, source='sketchfab', rank=0, world_size=1, **kwargs):
    """Download objects. Called by download.py as dataset_utils.download(metadata, **opt)."""
    output_dir = download_root or root
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # get objaverse annotations to resolve download URLs
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]

    # download objects
    file_paths = oxl.download_objects(
        annotations,
        download_dir=os.path.join(output_dir, 'raw'),
        save_repo_format='zip',
    )

    sha256_to_file_id = metadata.set_index('file_identifier')['sha256'].to_dict() \
        if 'file_identifier' in metadata.columns else {}

    downloaded = {}
    for file_id, local_path in file_paths.items():
        sha256 = sha256_to_file_id.get(file_id)
        if sha256 is None:
            # fall back: lookup by matching annotations
            match = annotations[annotations['fileIdentifier'] == file_id]
            if not match.empty:
                sha256 = match.iloc[0]['sha256']
        if sha256 is not None:
            downloaded[sha256] = os.path.relpath(local_path, output_dir)

    return pd.DataFrame(list(downloaded.items()), columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects',
                     no_file=False) -> pd.DataFrame:
    """Iterate over dataset instances and apply func(file_path, metadatum_dict).

    Called by dump_mesh.py, dump_pbr.py, dual_grid.py, voxelize_pbr.py, etc.

    Args:
        metadata:    DataFrame with at least sha256 and local_path columns.
        output_dir:  Base directory for resolving local_path. Can be None when no_file=True.
        func:        Callable(file_path, metadatum_dict) -> dict | None
        max_workers: Thread pool size. Defaults to cpu_count().
        desc:        tqdm description string.
        no_file:     When True, func is called as func(None, metadatum_dict) — used by
                     dual_grid.py and voxelize_pbr.py which read from mesh_dumps directly.
    """
    import tempfile
    import zipfile

    rows = metadata.to_dict('records')
    records = []
    max_workers = max_workers or os.cpu_count()

    def worker(metadatum):
        try:
            if no_file:
                return func(None, metadatum)

            local_path = metadatum.get('local_path', '')
            if local_path and local_path.startswith('raw/github/repos/'):
                path_parts = local_path.split('/')
                file_name = os.path.join(*path_parts[5:])
                zip_file = os.path.join(output_dir, *path_parts[:5])
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(tmp_dir)
                    file = os.path.join(tmp_dir, file_name)
                    return func(file, metadatum)
            else:
                file = os.path.join(output_dir, local_path) if (output_dir and local_path) else local_path
                return func(file, metadatum)
        except Exception as e:
            print(f"Error processing object {metadatum.get('sha256', '?')}: {e}")
            return None

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
             tqdm(total=len(rows), desc=desc) as pbar:
            for result in executor.map(worker, rows):
                if result is not None:
                    records.append(result)
                pbar.update()
    except Exception as e:
        print(f"Error during processing: {e}")

    return pd.DataFrame.from_records(records)
