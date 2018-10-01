from distutils.core import setup

setup(
    name='ResNext',
    version='0.1',
    packages=['ResNext',],
    license='pampam license',
    long_description=open('README.txt').read(),
    install_requires=[
        "certifi==2018.8.24",
		"numpy==1.15.2",
		"Pillow==5.3.0",
		"six==1.11.0",
		"torch==0.4.1.post2",
		"torchvision==0.2.1"],
)