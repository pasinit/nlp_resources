from setuptools import setup

setup(
    name='nlp_resources',
    version='1.0',
    # py_modules=["twenty_newsgroup"],
    packages = ["data_io", "nlp_models", "nlp_resources", "nlp_utils", "allennlp_mods"],
    url='https://github.com/tommy9114/nlp_resources',
    license='',
    author='tommaso',
    author_email='p.tommaso@gmail.com',
    description='',
    install_requires=["transformers==2.0", "deprecated", "torchtext"]#, "torch==1.3.1"]
)
