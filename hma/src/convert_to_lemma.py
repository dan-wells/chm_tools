import pandas as pd
import sys

input_data = pd.read_csv(sys.argv[1], sep='\t', header=None)
output_data = pd.read_csv(sys.argv[2], sep='\t', header=None)
fname = open(sys.argv[2] + '_noisy_suffix','w')
f_ip = input_data[2]
f_pos = output_data[1]
f_lemma = output_data[2]
assert len(f_ip) == len(f_pos) == len(f_lemma)
for ip, pos, lemma in zip(f_ip, f_pos, f_lemma):
	if len(ip) > 45:
		ip = ip[:45]
	fname.write(ip +'\t' + pos + '\t' + lemma + '\n')

fname.close()



