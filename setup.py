from setuptools import setup

install_reqs = open('requirements.txt')
lines = install_reqs.readlines()
reqs = [str(each_req) for each_req in lines]

setup(name='Identiying fusulinids',
      version='0.1',
      description='Identifying fusulinids using convolutional neural network',
      long_description = read_file('README.md'),
      author='Meng Chen',
      author_email='meng.chen03@gmail.com',
      url='https://github.com/biomchen/id_fusulinids',
      license='MIT',
      install_requires=reqs,
      keywords = 'fusulinids convolutional neural network')
