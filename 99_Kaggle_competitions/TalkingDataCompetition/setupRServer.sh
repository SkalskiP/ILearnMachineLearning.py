#!/usr/bin/env bash

# Installing trusted key on the server database
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
# Adding the repository to the machine
sudo add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'

# This is the set-up script for Google Cloud.
sudo apt-get update
sudo apt-get install r-base r-base-dev
# OpenSSL libraries
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install libssl-dev
sudo apt-get install libxml2 libxml2-dev