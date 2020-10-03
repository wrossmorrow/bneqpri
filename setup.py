from setuptools import setup

from bneqpri import __version__
from bneqpri.cli import header

setup(
	name='bneqpri',
	version=__version__,
	description='Fixed-Point Methods for Computing Equilibrium Prices',
	long_description=header,
	author='W. Ross Morrow',
	author_email='morrowwr@gmail.com',
	url='https://github.com/wrossmorrow/bneqpri',
	packages=['bneqpri'],
	scripts=['bin/bneqpri'],
	install_requires=['numpy']
)