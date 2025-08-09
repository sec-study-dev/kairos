# kairos

### Dataset

Due to GitHub storage space limitations, please download the data from this URL: https://drive.google.com/file/d/17gNByeqrbvUi2-VMPeChNaLgokN7jQ1r/view?usp=drive_link



### Pre-requisites

Install common software dependencies:

```python
pip3 install GitPython pandas tqdm scikit-learn
```

For GPU/CPU versions, dependencies can be installed using the following commands:

```python
pip3 install -r GPUCPU/requirements.txt
```

For the FPGA version, first connect the FPGA development board and follow these steps to set up the hardware and software environment:

1. Replace the origional files in `/Accelerated_Algorithmic_Trading/hw/pricingEngine` with the files in submitted sbm directory.

2. Merge sum sources:

   ```
   cd ../Accelerated_Algorithmic_Trading/hw/
   mv ./pricingEngine ./pricingEngine.bak
   cp -rf ../xilinx-acc-2021_submission/sbm/src/hw/pricingEngine ./
   ```

3. Build settings in ~/.bashrc: 

   ```
   source /opt/Xilinx/Vitis/2021.1/settings64.sh
   source /opt/xilinx/xrt/setup.sh
   export PLATFORM_REPO_PATHS='/opt/xilinx/platforms'
   export LM_LICENSE_FILE="~/Xilinx.lic"
   export XILINX_PLATFORM='xilinx_u50_gen3x16_xdma_201920_3'
   export DEVICE=${PLATFORM_REPO_PATHS}/${XILINX_PLATFORM}/${XILINX_PLATFORM}.xpfm
   export DM_MODE=DMA
   ```

4.  Build hardware and software:

   ```
   cd ../Accelerated_Algorithmic_Trading/build
   make clean
   sh ./buildall.sh
   ```

5. Build software:

   ```
   cd ../Accelerated_Algorithmic_Trading/sw/applications/aat/aat_shell_exe
   make allRunning Instructions
   ```



### Running Instructions

First, run the archiving node to provide services for Ethereum HTTP JSON-RPC on port 8545.

Then start the Hardhat client:

```
cd eth_clients/hardhat
bash setup_hardhat.sh
bash launch_hardhats.sh
```

