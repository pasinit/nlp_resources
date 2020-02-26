from setuptools import setup

setup(
    name='nlp_resources',
    version='1.0',
    packages = ["data_io", "nlp_models", "nlp_resources", "nlp_utils", "allennlp_mods"],
    url='https://github.com/tommy9114/nlp_resources',
    license='CC-BY-NC-4.0',
    author='tommaso',
    author_email='p.tommaso@gmail.com',
    description='',
    install_requires=["transformers==2.5.1", "deprecated", "torchtext", "torch>=1.2.0"],
)
