# This is an example call for indra.

import argparse
from indra import indra

# Define a parser.
PARSER = argparse.ArgumentParser(
    description="This is Vali, son of INDRA, generator of synthetic " +
    "weather time series. This function both 'learns' the structure of " +
    "data and samples from the learnt model. Both run modes need 'seed' " +
    "data, i.e., some input weather data.\r\n", prog='INDRA_VALI',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
PARSER.add_argument("--n_samples", type=int, default=10,
                    help="How many samples do you want out?")

ARGS = PARSER.parse_args()

n_samples = ARGS.n_samples

indra(train=1, station_code='gen', n_samples=10,
      path_file_in='gen/gen_iwec.epw',
      path_file_out='gen/gen_iwec_syn.epw',
      file_type='espr', store_path='gen', arma_params=[1, 1, 0, 0, 0])

for SAMPLE in range(0, n_samples):
    indra(train=0, station_code='gen',
          path_file_in='gen/gen_iwec.epw',
          path_file_out='gen/gen_iwec_syn_{:02d}.a'.format(SAMPLE),
          file_type='epw', store_path='gen')

print("I've run out of samples - call me again to get a fresh set.")