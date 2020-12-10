from setuptools import setup

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name="patchperformance",
    version="1.0",
    description="Calculate the patch performance on the fly while training your neural networks",
    url="https://github.com/IPMI-ICNS-UKE/patch-performance",
    download_url="https://github.com/IPMI-ICNS-UKE/patch-performance/tarball/master",
    author="Rüdiger Schmitz & Frederic Madesta",
    author_email="r.schmitz@uke.de, f.madesta@uke.de",
    license="GNU GPLv3",
    packages=["patchperformance"],
    zip_safe=True,
    include_package_data=True,
    tests_require=["pytest"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[],
    python_requires=">=3.6",
    extras_require={"torch": ["torch"], "tensorflow": ["tensorflow"]},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
