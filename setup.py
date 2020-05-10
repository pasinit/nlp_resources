from setuptools import setup

setup(
    name='nlp_resources',
    version='2.0',
    #package_dir = {'': 'nlp_resources'},
    packages = ["nlp_resources","nlp_resources.allen_data", "nlp_resources.data_io", "nlp_resources.allennlp_training_callbacks", 
         "nlp_resources.nlp_models", "nlp_resources.nlp_utils"],
    url='https://github.com/tommy9114/nlp_resources',
    license='CC-BY-NC-4.0',
    author='tommaso',
    author_email='p.tommaso@gmail.com',
    description='',
    install_requires=["transformers>=2.7.0", "deprecated", "torchtext", "torch", "allennlp==1.0.0rc4.dev20200505"],
)
