from setuptools import setup

setup(
    name='nlp_tools',
    version='2.0',
    #package_dir = {'': 'nlp_resources'},
    packages = ["nlp_tools","nlp_tools.allen_data", "nlp_tools.data_io", "nlp_tools.allennlp_training_callbacks", 
         "nlp_tools.nlp_models", "nlp_tools.nlp_utils"],
    url='https://github.com/tommy9114/nlp_tools',
    license='CC-BY-NC-4.0',
    author='tommaso',
    author_email='p.tommaso@gmail.com',
    description='',
    install_requires=["transformers==2.8.0", "deprecated", "allennlp==1.0.0rc4", "torchtext", "torch==1.4.0"],
)
