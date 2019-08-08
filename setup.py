from setuptools import setup

setup(
    name='nlp_resources',
    version='0.1.0',
    # py_modules=["twenty_newsgroup"],
    packages = ["nlp_resources", "nlp_models"],
    # package_dir={'': 'nlp_resources'},
    url='https://github.com/tommy9114/nlp_resources',
    license='',
    author='tommaso',
    author_email='p.tommaso@gmail.com',
    description='',
    install_requires=["pytorch_pretrained_bert==0.6.2"]
)
