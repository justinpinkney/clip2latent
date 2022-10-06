from setuptools import setup

setup(
    name='clip2latent',
    packages=['clip2latent'],
    version='1.0',
    description='Official code for clip2latent',
    author='Justin Pinkney',
    package_data = {'clip2latent': ['stylegan3/*']}
)
