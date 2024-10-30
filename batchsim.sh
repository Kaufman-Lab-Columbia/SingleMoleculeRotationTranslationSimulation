#!/bin/bash

for d in */
do

	cd $d
	pwd

	for dir in */
	do
		cd $dir
		nvcc RotationTranslationSimulation.cu -o RotationTranslationSim.exe
		./RotationTranslationSim.exe
		cd ..
	done

	cd ..

done

echo "Simulations Complete!"
