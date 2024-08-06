# HLDetectionTracking

## Authors
[Ahmet Agaoglu](https://github.com/Ahmet-Agaoglu), Nezih Topaloglu

## Description
This repository contains the source code for maritime horizon line detection and tracking. The theoretical background of this code is detailed in the paper "Fast Maritime Horizon Line Tracking with On-Off Region-of-Interest Control" by Ahmet Agaoglu and Nezih Topaloglu, currently under review.

## Files Included

The repository includes three Python files and two folders:

**main.py** processes all videos in a chosen dataset and obtains performance metrics. Run this file to execute the main functionality of the project.

**HLDA.py** contains the horizon line detection algorithm (HLDA) used in this study. The theory behind this algorithm is presented in the paper "An intensity-difference-based maritime horizon detection algorithm" by Nezih Topaloglu (available [here](https://link.springer.com/article/10.1007/s11760-024-03219-9)). This file can also be run separately to apply the HLDA to a single video file.

**helper.py** includes various utility functions, such as error estimation calculations, which support the main and HLDA files.

The folder **VIS_Onboard** includes the ground truth daha for the Singapore Maritime Dataset, On-Board Videos (subfolder: HorizonGTCorrected).

The folder **Buoy** includes the ground truth daha for the Buoy Dataset (subfolder: HorizonGTCorrected)

## Getting Started

### Prerequisites:
NumPy and OpenCV libraries should be installed.

### Installation:
1. Download the repository into your system.
2. Download the Singapore Maritime Dataset On-Board videos into `~/VIS_Onboard/Videos` via [this link](https://drive.google.com/file/d/0B43_rYxEgelVb2VFaXB4cE56RW8/view?resourcekey=0-67PrivAOYTGyWxAO_-2n1A)
3. Download the Buoy Dataset videos into `~/Buoy/Videos` via [this link](https://drive.google.com/file/d/0B43_rYxEgelVVngtMVBpWGFqckE/view?resourcekey=0-zBgpYCkkblxPZocaf8NU5w)

### Usage

How to run the code:
`python main.py`

Or to run the horizon line detection algorithm separately:
`python HLDA.py`

## Citation
If you use this code in your research, please cite the following paper:
```
@article{Agaoglu2024,
  title={Fast Maritime Horizon Line Tracking with On-Off Region-of-Interest Control},
  author={Ahmet Agaoglu and Nezih Topaloglu},
  journal={Under Review},
  year={2024}
}
```

If you use the horizon line detection algorithm provided in HLDA.py, please cite the following paper:

```
@Article{Topaloglu24,
author={Topaloglu, Nezih},
title={An intensity-difference-based maritime horizon detection algorithm},
journal={Signal, Image and Video Processing},
year={2024},
month={May},
day={06},
doi={10.1007/s11760-024-03219-9},
url={https://doi.org/10.1007/s11760-024-03219-9}
}
```
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

