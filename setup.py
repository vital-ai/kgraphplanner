from setuptools import setup, find_packages

setup(
    name='kgraphplanner',
    version='0.0.7',
    author='Marc Hadfield',
    author_email='marc@vital.ai',
    description='KGraph Planner',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vital-ai/kgraphplanner',
    packages=find_packages(exclude=["test"]),
    entry_points={

    },
    scripts=[

    ],
    package_data={
        '': ['*.pyi']
    },
    license='Apache License 2.0',
    install_requires=[
        'vital-ai-vitalsigns>=0.1.20',
        'vital-ai-aimp>=0.1.7',
        'vital-ai-haley-kg>=0.1.13',
        'kgraphagent>=0.0.1',
        'vital-agent-kg-utils>=0.1.4',
        'langchain-core==0.3.6',
        'langchain==0.3.1',
        'langchain-openai==0.2.1',
        'langgraph==0.2.29',
        'langgraph-checkpoint==1.0.13',
        'langsmith==0.1.129',
        'python-dotenv',
        'rich==13.7.1'
    ],
    extras_require={
        'dev': [
            'twine',
            'setuptools'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
