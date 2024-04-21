from setuptools import setup, find_packages

setup(
    name='overeasy',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A utility library for object detection tasks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/overeasy/overeasy',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
)