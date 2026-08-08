import os
import sys
from pathlib import Path
from typing import List, Tuple


HERE = Path(__file__).resolve().parent
PICO_TREE_INCLUDE = HERE / "include" / "pico_tree" / "src" / "pico_tree"


def get_build_flags() -> Tuple[List[str], List[str]]:
    compile_flags = ["-O3", "-std=c++17"]
    link_flags: List[str] = []

    # PyTorch wheels bundle libomp on macOS; loading a second runtime can crash.
    if sys.platform == "darwin" or os.environ.get("MEGANORM_DISABLE_OPENMP") == "1":
        return compile_flags, link_flags

    compile_flags.append("-fopenmp")
    link_flags.append("-fopenmp")

    return compile_flags, link_flags
