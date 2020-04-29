from setuptools import setup

setup(
    name='nlp_resources',
    version='1.1',
    packages = ["data_io", "to_be_updated/deprecated_allennlp_mods", "nlp_models", "nlp_utils"],
    url='https://github.com/tommy9114/nlp_resources',
    license='CC-BY-NC-4.0',
    author='tommaso',
    author_email='p.tommaso@gmail.com',
    description='',
    install_requires=["transformers>=2.5.1", "deprecated", "torchtext", "torch>=1.4.0"],
)
