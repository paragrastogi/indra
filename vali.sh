	#!/bin/bash

	# How many samples do you want?
	read -p 'How many samples do you want? ' n_samples

	echo 'Running for '$n_samples' samples'

	python indra.py --train 1 --station_code 'gen' --n_samples $n_samples --path_file_in 'gen/che_geneva.iwec.a' --path_file_out 'gen/gen_iwec_syn.a' --file_type 'espr' --store_path 'gen' --arma_params 1,1,0,0,0

	for SAMPLE in $(seq $n_samples)
	do
		echo $(printf 'gen/gen_iwec_syn_%02d.a' $SAMPLE)
		python indra.py --train 0 --station_code 'gen' --path_file_in 'gen/che_geneva.iwec.a' --path_file_out $(printf 'gen/gen_iwec_syn_%02d.a' $SAMPLE) --file_type 'espr' --store_path 'gen'
	done
