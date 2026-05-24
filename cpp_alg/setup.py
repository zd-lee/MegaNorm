from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os


here = os.path.dirname(os.path.abspath(__file__))

setup(
    name="cpp_alg",
    ext_modules=[
        CppExtension(
            name="cpp_alg_ext",
            sources=["patch_bfs.cpp", "connected_component.cpp", "ply_io.cpp", "split_patches.cpp", "fps_cpu.cpp"],
            include_dirs=[
                os.path.join(here, "include", "pico_tree", "src", "pico_tree")
            ],
            extra_compile_args={"cxx": ["-O3", "-std=c++17", "-fopenmp"]},
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

