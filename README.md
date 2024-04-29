# Insta360 autostitcher
A small tool that monitors a "dump" directory for `*.insv` files and stitches them into a 360 video.

Note that this tool uses Insta360 MediaSDK, which requires you to apply for the SDK.

I am not a software engineer so the code might be bad. Use it at your own risk. PRs are welcomed :)
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
