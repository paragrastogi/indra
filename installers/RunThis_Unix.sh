#!/bin/bash
# Ask the user for the name of their python package manager.
read -p 'Enter the name of your preferred package manager (in lower case): ' installer

echo $installer

if [ $installer = 'conda' ]
then
	conda install --file py_packages.txt
else
	pip install -r py_packages.txt
fi

echo "Finished!"
exit 
