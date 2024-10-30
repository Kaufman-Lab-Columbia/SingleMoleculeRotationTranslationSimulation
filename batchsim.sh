#!/bin/bash

for d in */
do

	cd $d
	pwd

	for dir in */
	do
		cd $dir
		nvcc RotTransSimulation_addCorrExch_FixShortExchTime_Sept2024.cu -o RotTransSim.exe
		./RotTransSim.exe
		cd ..
	done

	cd ..

done

echo "DID IT!"
