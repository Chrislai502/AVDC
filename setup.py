from setuptools import setup, find_packages

setup(
    name="avdc",
    version="0.1.0",
    description="A package for AVDC project",
    maintainer="Chris Lai",
    maintainer_email="cl@co.bot",  # Replace with your actual email
    python_requires=">=3.9, <4",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        # List your dependencies here, for example:
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "tqdm>=4.60.0",
        # Add more dependencies as needed
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            # Add more development dependencies if needed
        ]
    },
    entry_points={
        "console_scripts": [
            # Add any CLI scripts here if needed
            # "avdc-cli=avdc.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
)
