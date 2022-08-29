import re
from pathlib import Path
from typing import List

def get_all_checkpoints(checkpoint_dir: Path, checkpoint_prefix: str="global_step") -> List[str]:
    """
    Returns all saved checkpoints, sorted by recency (most recent first)
    """
    if checkpoint_dir.exists():
        checkpoints = sorted([item.name 
                                for item in checkpoint_dir.iterdir() 
                                if item.is_dir() and re.match(checkpoint_prefix + "\d+", item.name)], 
                             key=lambda x: int(re.match(checkpoint_prefix + "(\d+)", x).groups()[0]), reverse=True)
        return checkpoints
    else:
        return []