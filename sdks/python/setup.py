from setuptools import setup, find_packages

setup(
    name="databridge-client",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "httpx",
        "pyjwt",
    ],
    python_requires=">=3.7",
)
