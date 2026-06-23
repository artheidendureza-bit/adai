"""
Custom build extension for adAI C++ backend
"""
import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Custom build_ext to build C++ extension with CMake"""
    
    def run(self):
        # Build the C++ extension using CMake
        cpp_backend_dir = Path(__file__).parent
        build_dir = cpp_backend_dir / "build"
        
        # Create build directory
        build_dir.mkdir(exist_ok=True)
        
        # Configure with CMake
        cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DADAICPP_BUILD_TESTS=OFF",
            "-DADAICPP_BUILD_PYTHON=ON",
            "-DADAICPP_ENABLE_CUDA=OFF",
        ]
        
        # Configure
        subprocess.check_call(
            ["cmake", str(cpp_backend_dir)] + cmake_args,
            cwd=str(build_dir)
        )
        
        # Build
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release"],
            cwd=str(build_dir)
        )
        
        # Copy the built extension to the source directory
        import shutil
        
        # Find the built extension
        if sys.platform == "win32":
            ext_pattern = "adaicpp_py.pyd"
        else:
            ext_pattern = "adaicpp_py*.so"
        
        built_exts = list(build_dir.rglob(ext_pattern))
        if built_exts:
            src_ext = built_exts[0]
            dst_ext = cpp_backend_dir.parent / "src" / "adAI" / src_ext.name
            shutil.copy(src_ext, dst_ext)
            print(f"Copied C++ extension to {dst_ext}")
        else:
            print(f"Warning: Could not find built extension matching {ext_pattern}")
        
        # Run the standard build_ext
        super().run()


if __name__ == "__main__":
    BuildExt(None).run()
