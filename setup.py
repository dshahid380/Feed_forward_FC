from distutils.core import setup, Extension

neuron_module = Extension('neuron', sources = ['neuron.c'])

setup(name='neuron',
      version='0.1.0',
      description='Neural Networks for Python in C',
      ext_modules=[neuron_module])
