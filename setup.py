from setuptools import setup, find_packages

req_file = open("requirements.txt", "r")
reqs = [line.rstrip('\n') for line in req_file]


setup(
    name='mia',
    version='1.0',
    scripts='mia',
    author='Christian Eschen',
    author_email='christian_eschen@hotmail.com',
    packages=find_packages(),
    description='mia - medical image analysis (angiographies)',
    long_description='README.md',
    long_description_content_type="text/markdown",
    url='https://github.com/ChristianEschen/mia',
    license='LICENSE.txt',
    install_requires=reqs,
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    )
