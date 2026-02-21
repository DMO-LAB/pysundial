from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import sysconfig

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = 'Debug' if self.debug else 'Release'

        python_include_dir = sysconfig.get_paths()['include']
        python_library = os.path.join(sysconfig.get_config_var('LIBDIR'), 'libpython' + sysconfig.get_python_version() + '.so')

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DPython_EXECUTABLE={sys.executable}',
            f'-DPYTHON_INCLUDE_DIR={python_include_dir}',
            f'-DPYTHON_LIBRARY={python_library}'
        ]

        build_args = ['--config', cfg, '--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

setup(
    name='SundialsPy',
    version='0.1.0',
    author='Eloghosa Ikponmwoba',  
    author_email='eloghosaefficiency@gmail.com', 
    description='Python bindings for SUNDIALS', 
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='.'),
    package_dir={'SundialsPy': 'SundialsPy'},
    ext_modules=[CMakeExtension('SundialsPy._SundialsPy')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=[
        'numpy',
        # Add any other Python dependencies here
        ],
    python_requires='>=3.7', # Or your minimum Python version
)