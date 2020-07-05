#!/usr/bin/env bash

cd grouping
sh tf*abi.sh
cd ..

cd interpolation
sh tf*abi.sh 
cd ..

cd sampling
sh tf*abi.sh
cd ..
