from setuptools import setup, find_packages

setup(
    name='kgraphplanner',
    version='0.0.18',
    author='Marc Hadfield',
    author_email='marc@vital.ai',
    description='KGraph Planner',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vital-ai/kgraphplanner',
    packages=find_packages(exclude=[
        "test", "test.*",
        "test_scripts", "test_scripts.*",
        "test_scripts_planner", "test_scripts_planner.*",
        "kgraphplanner_orig", "kgraphplanner_orig.*",
        "notes", "notes.*",
        "planning", "planning.*",
        "docs", "docs.*",
    ]),
    entry_points={

    },
    scripts=[

    ],
    package_data={
        '': ['*.pyi']
    },
    license='Apache License 2.0',
    install_requires=[
        'vital-ai-vitalsigns>=0.1.32',
        'vital-ai-aimp>=0.1.16',
        'vital-ai-haley-kg>=0.1.24',
        'langchain-core>=1.2.9',
        'langchain-openai>=1.1.7',
        'langgraph>=1.0.8',
        'langgraph-checkpoint>=4.0.0',
        'langsmith',
        'vital-agent-kg-utils>=0.1.7',
        # 'langchain-core>=0.3.21',
        # 'langchain==0.3.9',
        # 'langchain-openai==0.2.1',
        # 'langgraph==0.2.53',
        # 'langgraph-checkpoint==2.0.7',
        # 'langsmith',
        # 'pydantic>=2.8.2',
        'python-dotenv',
        'rich>=13.7.1'
    ],
    extras_require={
        'dev': [
            'twine',
            'setuptools',
            'pygraphviz',
            'nest-asyncio',
            'pyppeteer'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
