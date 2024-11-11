# Network Optix 2024 EVS Live Coding Session

## Description

This repository contains the source code for a custom NX AI Manager postprocessor and a tiny Python HTTP server that servers a live webpage displaying the results from a Gauge model trained on [Teachable Machine](https://teachablemachine.withgoogle.com/train).

For more information on deploying a Teachable Machine model to an edge device running an NX Server, visit the [NX documentation](https://nx.docs.scailable.net/for-data-scientists/importing-models/from-teachable-machine).

To learn more about custom NX AI Manager postprocessors, see the [Scailable Integration SDK](https://github.com/scailable/sclbl-integration-sdk).

## Requirements

- Raspberry Pi 4 or 5
- Webcam
- NX Meta Server
- NX Meta Client
- Software:
    - Cmake
    - Python 3
```bash
sudo apt install cmake
sudo apt install python3.11-venv
sudo apt install python3.11-dev
```
(If you have Python3.12 installed, you'll need to install `python3.12-venv` & `python3.12-dev` instead)

## Instructions

1. Follow the [NX AI Manager installation documentation](https://nx.docs.scailable.net/).
2. Choose the ARM64 version of the NX Server: [Download here](https://updates.networkoptix.com/metavms/6.0.0.38488/arm/metavms-server-6.0.0.38488-linux_arm64-beta.deb).
3. Clone this repository to your Raspberry Pi.
4. Navigate to the repository directory:
    ```bash
    cd <repository-directory>
    ```
5. Run the build script:
    ```bash
    ./1_build_post_processor.sh
    ```
6. Start the API server:
    ```bash
    ./2_start_api.sh
    ```
7. Go to port :8888 at the ip address of the Raspberry Pi with a browser (nicest is to use the browser within Nx Client).
    
https://github.com/scailable/nx-evs-2024-demo/assets/95014/fa2cb534-9c34-425b-8dd3-298834821861

