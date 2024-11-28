from setuptools import setup, find_packages

setup(
    name='kgraphplanner',
    version='0.0.12',
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
        'vital-ai-vitalsigns>=0.1.23',
        'vital-ai-aimp>=0.1.7',
        'vital-ai-haley-kg>=0.1.13',
        'kgraphagent>=0.0.1',
        'vital-agent-kg-utils>=0.1.5',
        'langchain-core==0.3.21',
        'langchain==0.3.9',
        'langchain-openai==0.2.1',
        'langgraph==0.2.53',
        'langgraph-checkpoint==2.0.7',
        'langsmith',
        'python-dotenv',
        'rich==13.7.1',
        'pydantic>=2.8.2'
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
