from setuptools import setup

setup(name='PyGPR',
      version='0.1',
      description='Scalable Gaussian Process Regression implemented in python',
      url='http://github.com/sarathsrinivas/PyGPR.git',
      author='Sarath Srinivas S',
      author_email='srinix@pm.me',
      license='MIT',
      packages=['PyGPR'],
      install_requires=['torch', 'numpy', 'scipy'],
      zip_safe=False)
