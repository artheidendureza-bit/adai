from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="adaiml-tools",
    version="0.1.1",
    description="A beginner-friendly AI library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Artheiden",
    author_email="artheidendureza@gmail.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=["numpy>=1.24"],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)