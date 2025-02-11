import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="plot_posthoc_test",
    version="0.0.1",
    author="Carlos Johnson-Cruz",
    author_email="cjohnsoncruz@gmail.com",
    description="Package for performing and plotting posthoc tests, for use with seaborn/matplotlib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cjohnsoncruz/plot_posthoc_test",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=open("requirements.txt").readlines(),
    python_requires='>=3.8',
)