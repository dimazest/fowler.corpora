import os.path
import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = 'test'
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


dirname = os.path.dirname(__file__)
long_description = (
    open(os.path.join(dirname, 'README.rst')).read()
)


setup(
    name='fowler.corpora',
    version='0.1',
    description='',
    long_description=long_description,
    # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Topic :: Utilities',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    keywords='',
    author='Dmitrijs Milajevs',
    author_email='dimazest@gmail.com',
    url='https://github.com/dimazest/fowler.corpora/',
    license='MIT',
    packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
    namespace_packages=['fowler'],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'fowler.switchboard',
        'google-ngram-downloader',
        'ipython',
        'jinja2',
        'joblib',
        'matplotlib',
        'more-itertools',
        'nltk',
        'numexpr',
        'numpy',
        'openpyxl<1.9999',  # Pandas requires vresion < 2.0
        'opster',
        'pandas',
        'progress',
        'py',
        'raven',
        'scikit-learn',
        'scipy',
        'seaborn',
        'setuptools',
        'six',
        'tables',
        'tornado',
        'xlwt-future',
        'zope.cachedescriptors',
    ],
    entry_points={
        'console_scripts': [
            'corpora = fowler.corpora.main:dispatch',
        ],
    },
    tests_require=['pytest>=2.4.2', 'pytest-bdd'],
    cmdclass={'test': PyTest},

)
