from setuptools import setup

setup(
    name='pmf',
    version='1.0.0',
    packages=['pmf', 'pmf.data', 'pmf.model', 'pmf.vi'],
    install_requires=[
    	'numpy==1.18.4',
	    'tqdm==4.62.3',
	    'scipy==1.4.1',
	    'scikit_learn==1.0.2'
    ],
    license='MIT'
)

