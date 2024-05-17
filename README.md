# KAMERA: **K**itware's Image **A**cquisition **M**anag**ER** and **A**rchiver

Welcome to the official repository for KAMERA, an open-source software platform for data collection, management, and analysis. Developed by Kitware in collaboration with NOAA's Marine Mammal Laboratory, KAMERA utilizes synchronized data streams and deep learning to detect and map key marine species like polar bears and ice-associated seals in Arctic and sub-Arctic regions.

## About KAMERA

KAMERA integrates cutting-edge technology with environmental research efforts, offering tightly synchronized data streams and real-time deep learning models to facilitate in-depth data analysis and efficient surveying of marine mammals. This tool is designed to assist researchers, conservationists, and data scientists in collecting and analyzing large-scale geographical and environmental data, enhancing the understanding and conservation of marine ecosystems.

## Features

- **Data Synchronization**: Integrates data from multiple sensors and cameras, ensuring synchronized data collection.
- **Real-time Analysis**: Implements deep learning models in [VIAME] (https://github.com/VIAME/VIAME) for real-time detection and analysis of marine life.
- **Hardware Integration**: Supports a variety of camera types and an Inertial Navigation System (INS) for precise data capture.
- **Open-Source Flexibility**: Licensed under Apache 2.0, allowing for broad modification, distribution, and use.

## Installation

```bash
git clone https://github.com/Kitware/kamera.git
cd kamera
# For the pure post-processing and generating flight summary, you can install
# the requirements in requirements.txt, or use the provided dockerfile
make build-postflight
# Builds the core docker images for use in the onboard sytems
make build-core
# if using VIAME for the DL detectors
make build-viame
```
Note that these images take up a large amount of disk space, especially the VIAME image which is 30Gb, and it can take several hours to builds. The core images are faster and lighter weight.

## Usage

Please let us know if we can help integrate this software into your solution!

## Contributing

We welcome contributions!

## Partners and Acknowledgements
KAMERA was developed in collaboration with:

    - NOAA Marine Mammal Laboratory
    - University of Washington
We thank all contributors who have helped in developing KAMERA, with special thanks to Mike McDermott and Matt Brown.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE] () for details.

## Contact
For further information, support, or collaboration inquiries, please contact adam.romlein@kitware.com

We hope KAMERA will empower your research and conservation efforts, and we look forward to seeing how you will use this system.
