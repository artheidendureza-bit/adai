from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return
        
        import subprocess
        
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DADAICPP_BUILD_PYTHON=ON',
            '-DADAICPP_BUILD_TESTS=OFF',
        ]
        
        build_args = ['--config', 'Release']
        
        if sys.platform == 'win32':
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                'Release', extdir)]
            build_args += ['--config', 'Release']
        
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, 
                            cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                            cwd=self.build_temp)

setup(
    name='adaicpp',
    version='0.1.0',
    author='adAI',
    description='adAI C++ Backend - High-performance tensor operations',
    ext_modules=[CMakeExtension('adaicpp', sourcedir='..')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.6',
)
