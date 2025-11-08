from setuptools import find_packages, setup
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
with open(readme_file, "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements if available, otherwise use inline list
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]
else:
    # Fallback to inline requirements for building from sdist
    requirements = [
        "numpy>=1.24.0,<2.0.0",
        "elasticsearch>=8.11.0,<9.0.0",
        "gTTS>=2.4.0,<3.0.0",
        "openai>=1.3.0,<2.0.0",
        "anthropic>=0.7.0,<1.0.0",
        "google-generativeai>=0.3.0,<1.0.0",
        "requests>=2.31.0,<3.0.0",
        "httpx>=0.25.0,<1.0.0",
        "pydub>=0.25.1,<1.0.0",
        "librosa>=0.10.0,<1.0.0",
        "soundfile>=0.12.0,<1.0.0",
        "torch>=2.1.0,<3.0.0",
        "torchaudio>=2.1.0,<3.0.0",
        "transformers>=4.35.0,<5.0.0",
        "datasets>=2.14.0,<3.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "loguru>=0.7.0,<1.0.0",
        "python-dotenv>=1.0.0,<2.0.0",
        "click>=8.1.0,<9.0.0",
        "rich>=13.7.0,<14.0.0",
        "tqdm>=4.66.0,<5.0.0",
    ]

setup(
    name="cogitura",
    version="0.1.0",
    author="TheusHen",
    author_email="",
    description="Projeto de pesquisa: IAs criando outras IAs através de treinamento de reconhecimento de voz",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheusHen/Cogitura",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cogitura=cogitura.cli:main",
        ],
    },
)
