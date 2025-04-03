from setuptools import setup, find_packages

setup(
    name='climada_petal_urban_flood',
    version='0.1.0',
    description='Urban flood module for CLIMADA platform',
    author='Nazila Sepehrkiya',
    author_email='env.phd.2022@gmail.com',
    url='https://github.com/sepehrkiya/climada_petal_urban_flood,
    packages=find_packages(),
    install_requires=[
        'climada>=6.0.1',
    climada>=6.0.1
    numpy>=1.26.4
    pandas>=2.2.3
    geopandas>=1.0.1
    rasterio>=1.4.3
    scipy>=1.15.2
    matplotlib>=3.10.0
    shapely>=2.0.7
    osmnx>=2.0.2
    pytest>=8.3.5
    sphinx>=8.2.3
    cdsapi>=0.7.5
    elevation>=1.1.3
    slope>=0.1.0
    datetime>=5.5
    requests>=2.32.3
    pathlib>=1.0.
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.11.11',
        'Topic :: Environmental Planning :: Atmospheric Science',
    ],
    package_data={
        'climada_petal_urban_flood': ['data/hazard/*.nc', 'data/exposure/*.shp , https://cds.climate.copernicus.eu/datasets'],
    },
)
