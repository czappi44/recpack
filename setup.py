from pathlib import Path
from setuptools import setup, find_packages

SHORT_DESCRIPTION = """Python package for Top-N recommendation based on implicit feedback data."""

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="recpack",
    version="0.3.1.dev1",
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.2, ==1.*",
        "scipy>=1.6.0, ==1.*",
        "scikit-learn>=1.1.1, ==1.*",
        "pandas>=1.2.1, ==1.*",
        "PyYAML>=5.4.1, ==5.*",
        "torch>=1.9.0, ==1.*",
        "tqdm>=4.46.0, ==4.*",
        "dataclasses==0.6",
    ]
    + ["pytest>=6.2.4, ==6.*", "pytest-cov>=2.12.1, ==2.*"],
    entry_points={},
    description=SHORT_DESCRIPTION,
    # long_description=long_description,
    url="https://gitlab.com/adrem-recommenders/recpack",
    project_urls={"documentation": "https://recpack.froomle.ai"},
)
