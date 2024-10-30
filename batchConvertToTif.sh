#!/bin/bash

for d in */
do
	cd $d

	for dir in */
	do
		cd $dir

		for direc in */
		do
			cd $direc
			pwd
			python //MODIFY_PATH//SimOut_to_Tif.py
			cd ..
		done
		
		cd ..

	done

	cd ..

done

echo "DID IT!"
