import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="plot_posthoc_test-pkg-your-username",
    version="0.0.1",
    author="Carlos Johnson-Cruz",
    author_email="cjohnsoncruz@gmail.com",
    description="Package for performing and plotting posthoc tests, for use with seaborn/matplotlib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)