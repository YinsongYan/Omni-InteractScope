"""Materialize the dataset from Hugging Face repository revisions."""
from pathlib import Path
import shutil


def download_dataset(repo_id, revision, destination):
    """Copy the selected dataset revision from the Hub cache to a local directory.

    Current revisions store data below ``data/Omni-InteractScope``. Older
    revisions may store it at the root. The destination always receives dataset
    contents directly, so callers keep the same ``--dataset`` semantics.
    """
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=repo_id, repo_type='dataset', revision=revision))
    nested = snapshot / 'data' / 'Omni-InteractScope'
    source = nested if (nested / 'annotations.json').is_file() else snapshot
    if not (source / 'annotations.json').is_file():
        raise ValueError('The selected repository revision does not contain Omni-InteractScope annotations.')
    source = source.resolve()
    destination = Path(destination).expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('The dataset destination must be outside the Hugging Face snapshot directory.')
    # Follow Hub-cache symlinks and create independent files in the destination.
    shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=False)
    return destination
