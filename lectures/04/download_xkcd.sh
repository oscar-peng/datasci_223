#!/bin/bash

# Create media directory if it doesn't exist
mkdir -p media

# Download XKCD comics
wget https://imgs.xkcd.com/comics/extrapolating.png -O media/extrapolating.png
wget https://imgs.xkcd.com/comics/correlation.png -O media/correlation.png
wget https://imgs.xkcd.com/comics/linear_regression.png -O media/linear_regression.png
wget https://imgs.xkcd.com/comics/machine_learning.png -O media/machine_learning.png

# Add attribution text
echo "XKCD Comics (https://xkcd.com) by Randall Munroe" > media/attribution.txt
echo "Used under Creative Commons Attribution-NonCommercial 2.5 License" >> media/attribution.txt 