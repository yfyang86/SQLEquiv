# setup.py
"""Setup script for sql-equivalence package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='sql-equivalence',
    version='0.1.0',
    author='Yifan Yang',
    author_email='yifan.yang@transwarp.io',
    description='A comprehensive library for analyzing SQL query equivalence',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://gitlab.transwarp.io/llm/tools/sql-equivalence',
    project_urls={
        'Bug Tracker': 'https://gitlab.transwarp.io/llm/tools/sql-equivalence/issues',
        'Source Code': 'https://gitlab.transwarp.io/llm/tools/sql-equivalence',
    },
    license='MIT',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'docs']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Database',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=[
        'sqlglot>=11.0.0',
        'networkx>=2.8',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
        'ml': [
            'torch>=1.10.0',
            'transformers>=4.20.0',
            'sentence-transformers>=2.2.0',
        ],
        'viz': [
            'matplotlib>=3.5.0',
            'graphviz>=0.20',
            'plotly>=5.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'sql-equiv=sql_equivalence.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'sql_equivalence': [
            'data/*.json',
            'data/*.yaml',
        ],
    },
    keywords=[
        'sql', 'query', 'equivalence', 'database', 
        'relational algebra', 'graph', 'embedding',
        'query optimization', 'sql analysis'
    ],
)