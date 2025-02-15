from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

import numpy as np

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy
        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())
        
        # Call original build_ext command
        build_ext.run(self)


_integration = Extension('ConvolutionMethods._integration',
                    sources = ['ConvolutionMethods/_integration.cpp'],
                    include_dirs = [np.get_include()],
                    extra_compile_args=['-O3'])
                    

_convolution = Extension('ConvolutionMethods._convolution',
                    sources = ['ConvolutionMethods/_convolution.cpp',
                               'ConvolutionMethods/convolution.cpp'],
                    include_dirs = [np.get_include()],
                    extra_compile_args=['-O3'])
                 

setup(
    name='NonlinearFunctionalConnectivity',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    ext_modules = [_integration,_convolution],
    author='Gabriel Marghoti',
    author_email='gabrielmarghoti@gmail.com',
    description='Convolution kernel fitting for complex systems dynamics.',
    url='https://github.com/GabrielMarghoti/NonlinearFunctionalConnectivity',
)