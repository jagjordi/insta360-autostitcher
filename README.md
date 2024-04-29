# Insta360 autostitcher
## Usage
1. Get the MediaSDK by applying [here](https://www.insta360.com/sdk/home)
2. Extract and copy the `libMediaSDK-dev_2.0-0_amd64_ubuntu18.04.deb` package in the repo folder
3. Build the docker image
```bash
docker build . -t insta360-autostitcher
```
4. Run it
```bash
docker run -it -v /PATH/TO/YOUR/FILES:/media insta360-autostitcher
```
