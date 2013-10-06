from setuptools import setup, find_packages


version = '0.1'


setup(
    name='fowler.corpora',
    version=version,
    description='',
    long_description='''''',
    # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[],
    keywords='',
    author='Dmitrijs Milajevs',
    author_email='dimazest@gmail.com',
    url='',
    license='',
    packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
    namespace_packages=['fowler'],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy',
        'opster',
        'setuptools',
        'nltk',
    ],
    entry_points={
        'console_scripts': [
            'corpora = fowler.corpora.main:dispatch',
        ],
    },
)
