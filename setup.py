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

# setup(
#     name='sundials_py',
#     version='0.1.0',
#     author='Your Name',
#     author_email='your.email@example.com',
#     description='Python bindings for SUNDIALS solvers',
#     long_description='',
#     ext_modules=[CMakeExtension('_sundials_py')],
#     packages=['sundials_py'],
#     package_dir={'_sundials_py': 'sundials_py'},
#     cmdclass=dict(build_ext=CMakeBuild),
#     zip_safe=False,
#     python_requires='>=3.6',
# )

setup(
    name='sundials_py',
    version='0.1.0',
    author='Eloghosa',  # Fill this in
    author_email='eloghosaefficiency@gmail.com', # Fill this in
    description='Python bindings for SUNDIALS', # Fill this in
    # long_description=open('README.md').read(), # Optional: Uncomment if you have a README
    # long_description_content_type='text/markdown', # Optional
    packages=find_packages(where='.'), # Automatically find 'sundials_py' package
    package_dir={'': '.'},             # Indicate packages are under the root directory
    ext_modules=[CMakeExtension('sundials_py._sundials_py')], # Correct extension name
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=[
        'numpy',
        # Add any other Python dependencies here
        ],
    python_requires='>=3.7', # Or your minimum Python version
)