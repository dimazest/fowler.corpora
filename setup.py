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
        # import here, cause outside the eggs aren't loaded
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
        'blosc',
        'chrono',
        'colored',
        'docutils',
        'eventlet',
        'execnet',
        'fowler.switchboard',
        'gensim',
        'google-ngram-downloader',
        'ipython',
        'jinja2',
        'joblib',
        'matplotlib',
        'more-itertools',
        'notebook',
        'numexpr',
        'numpy',
        'openpyxl',
        'opster',
        'pandas',
        'progress',
        'py',
        'pygments',
        'raven',
        'scikit-learn',
        'scipy',
        'seaborn',
        'setuptools',
        'six',
        'tables',
        'tornado',
        'XlsxWriter',
        'xlwt-future',
        'zope.cachedescriptors',

        'nltk>=3.0.0',  # and it's dependencies
        'twython',
    ],
    entry_points={
        'console_scripts': [
            'corpora = fowler.corpora.main:dispatch',
        ],
        'fowler.corpus_readers': [
            'brown = fowler.corpora.bnc.readers:Brown',
            'bnc = fowler.corpora.bnc.readers:BNC',
            'bnc-ccg = fowler.corpora.bnc.readers:BNC_CCG',
            'dep-parsed-ukwac = fowler.corpora.bnc.readers:UKWAC',

            'simlex999 = fowler.corpora.bnc.readers:SimLex999',
            'men = fowler.corpora.bnc.readers:MEN',

            'gs11 = fowler.corpora.bnc.readers:GS11',
            'gs12 = fowler.corpora.bnc.readers:GS12',
            'ks13 = fowler.corpora.bnc.readers:KS13',
            'phraserel = fowler.corpora.bnc.readers:PhraseRel',
        ],
    },
    tests_require=['pytest>=2.4.2', 'pytest-bdd', 'pytest-cov'],
    cmdclass={'test': PyTest},

)
