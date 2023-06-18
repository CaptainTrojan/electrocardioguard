import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='dgn_mlflow_logger',
    version='0.0.5',
    author='Michal Seják',
    author_email='sejakm@ntis.zcu.cz',
    description='MLFLow remote tracking for Diagnome',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Diagnome/dgn_mlflow_logger',
    license='MIT',
    packages=['dgn_mlflow_logger'],
    install_requires=['mlflow', 'tensorflow', 'nvidia-ml-py3', 'python-dotenv', 'torchview', 'torch',
                      'pytorch_lightning', 'graphviz', 'pandas'],
)
