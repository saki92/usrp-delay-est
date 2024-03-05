# USRP Delay Estimator
This tool measures the RTT of the signal sent and received on a USRP device.

## Hardware Setup
The Tx and Rx ports of the channel under test must be connected using attenuators or antennas must be attached.

## Software Setup
1. Download the repository with `git clone https://github.com/saki92/usrp-delay-est.git`
2. Build the USRP tool with
```
cd usrp-delay-est
mkdir build
cd build
cmake ../
make
```
3. Run the python script with `python3 usrp-delay.py`

## Pre-requisites
`boost`
`uhd`
`numpy`
`argparse`
`matplotlib`
