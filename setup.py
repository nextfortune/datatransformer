import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datatransformer", # Replace with your own username
    version="0.0.6",
    author="Tim Su",
    author_email="omg80827@gmail.com",
    description="A package for data transformation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.owdev.net/Jimmy/DataTransformer.git",
    project_urls={
        "Bug Tracker": "https://git.owdev.net/Jimmy/DataTransformer/-/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.4.1"
    ]
)
