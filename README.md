# HLDetectionTracking

## Authors
[Ahmet Agaoglu](https://github.com/Ahmet-Agaoglu), Nezih Topaloglu

## Description
This repository contains the source code for maritime horizon line detection and tracking. The theoretical background of this code is detailed in the paper "Fast Maritime Horizon Line Tracking with On-Off Region-of-Interest Control" by Ahmet Agaoglu and Nezih Topaloglu, currently under review.

## Files Included

The repository includes MATLAB files.

**main.m** Run this file to execute the main functionality of the project.

**HLDA.m** contains the horizon line detection algorithm (HLDA) used in this study. The theory behind this algorithm is presented in the paper Agaoglu, A., Topaloglu, N. Dynamic region of interest generation for maritime horizon line detection using time series analysis. Vis Comput (2025). https://doi.org/10.1007/s00371-024-03767-8

**error_metric.m** calculates the error metric.



## Getting Started

### Prerequisites:
MATLAB should be installed.

### Installation:
1. Download the repository into your system.
2. Download the Singapore Maritime Dataset On-Board videos into `~/VIS_Onboard/Videos` via [this link](https://drive.google.com/file/d/0B43_rYxEgelVb2VFaXB4cE56RW8/view?resourcekey=0-67PrivAOYTGyWxAO_-2n1A)
3. Download the Buoy Dataset videos into `~/Buoy/Videos` via [this link](https://drive.google.com/file/d/0B43_rYxEgelVVngtMVBpWGFqckE/view?resourcekey=0-zBgpYCkkblxPZocaf8NU5w)


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

If you use the horizon line detection algorithm provided in HLDA.m, please cite the following paper:

```
﻿@Article{Agaoglu2025,
author={Agaoglu, Ahmet
and Topaloglu, Nezih},
title={Dynamic region of interest generation for maritime horizon line detection using time series analysis},
journal={The Visual Computer},
year={2025},
month={Jan},
day={08},
issn={1432-2315},
doi={10.1007/s00371-024-03767-8},
```
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

