import argparse
import utils

parser = argparse.ArgumentParser(
                    prog='Denovo-frag',
                    description='Generate ligands using fragment',
                    )

parser.add_argument('-c', '--count', type=utils.parse_cmp_coordinates, required=True,
                    help='number of ligands to generate')

args = parser.parse_args()

for k, v in args.__dict__.items():
    print(k, ":", v)