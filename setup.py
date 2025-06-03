import setuptools

setuptools.setup(
    name="process_solidsuspension",
    version="0.0.1",
    description="Tools to process images for solid suspension",
    long_description="Tools to process images solid suspension. The code will predict the particle size distribution and concentration of solid particles in a suspension using image processing techniques.",
    url="",
    author="Weheliye Hashi",
    author_email="Weheliye.Weheliye@oribiotech.com",
    license="MIT",
    packages=setuptools.find_packages(),
    zip_safe=False,
    entry_points={
         'console_scripts': [
            'ss_gui=process_solidsuspension.ss_gui:main',
            'ss_main_processing=process_solidsuspension.Process_main_images_GUI:main',
        ],
    },
)