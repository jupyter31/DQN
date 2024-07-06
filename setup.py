from setuptools import setup, find_packages

setup(
    name='rl',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gym',
        'numpy',
        'matplotlib',
        'pyglet',
        'torch'
    ],
    entry_points={
        'console_scripts': [
            'rl-main = q_learning.main:main',
        ],
    },
)
