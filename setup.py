import setuptools


setuptools.setup(
    name="ynot",
    version="0.1",
    author="gully",
    author_email="igully@gmail.com",
    description="Forward Modeling 2D Echellograms",
    long_description="Experimental Data reduction for 2D astronomical echellograms",
    long_description_content_type="text/markdown",
    url="https://github.com/gully/ynot",
    install_requires=["numpy", "scipy", "torch"],
    packages=setuptools.find_packages(where="src"),
    package_dir={"":"src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
