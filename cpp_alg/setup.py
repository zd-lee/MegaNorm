from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

from build_config import PICO_TREE_INCLUDE, get_build_flags


here = Path(__file__).resolve().parent
compile_flags, link_flags = get_build_flags()

if not PICO_TREE_INCLUDE.is_dir():
    raise RuntimeError(
        "pico_tree is missing; run `git submodule update --init --recursive`"
    )

setup(
    name="cpp_alg",
    packages=["cpp_alg"],
    package_dir={"cpp_alg": str(here)},
    ext_modules=[
        CppExtension(
            name="cpp_alg.cpp_alg_ext",
            sources=[
                str(here / "patch_bfs.cpp"),
                str(here / "connected_component.cpp"),
                str(here / "ply_io.cpp"),
                str(here / "split_patches.cpp"),
                str(here / "fps_cpu.cpp"),
            ],
            include_dirs=[str(PICO_TREE_INCLUDE)],
            extra_compile_args={"cxx": compile_flags},
            extra_link_args=link_flags,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
