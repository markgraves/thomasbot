import argparse
import os, sys
import subprocess
from subprocess import call, check_output

def generate_sample(vsize=1024, words=10000, temperature=1, cuda=False, outf=None):
    gen_cmd = 'python topicbot/generate.py --checkpoint results/torch-models/aquinas_summa_full_bpe1024s_beospqa_l2_s256_emb256_lrd2_bptt200_e300.pt --bpemodel results/bpe-model/aquinas_summa_full_bpe1024s.model --words {words} --temperature {temp}'.format(
        words=words, temp=temperature)
    out_file ='results/generated-text/aquinas_summa_full_bpe1024s_beospqa_l2_s256_emb256_lrd2_bptt200_e300_gentemp{strtemp}_w{words}.txt'.format(
        words=words, strtemp=str(temperature).replace('.', 'd'))
    gen_cmd += ' --outf ' + out_file
    if cuda:
        gen_cmd += ' --cuda'
    print(gen_cmd)
    call(gen_cmd, shell=True)
    length = int(check_output('wc -w ' + out_file, shell=True).split()[0])
    misspelled = int(check_output('enchant -l %s|grep -v "eos"|wc -l' % out_file, shell=True).strip())
    misspelled_freq = float(misspelled) / float(length)
    print(out_file)
    print('words %s; misspelled %s; freq misspelled %s' % (length, misspelled, misspelled_freq))
    if outf:
        outf.write('\t'.join(str(x) for x in [vsize, words, temperature, length, misspelled, misspelled_freq, out_file]) + '\n')

def walk_samples(cuda=False, outf=None):
    for words in [5000, 100000]:
        for temperature in [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # for vsize in ['256', '512', '1k']:
            generate_sample(1024, words, temperature, cuda=cuda, outf=outf)


if __name__ == "__main__":
    with open('results/out-walk-temp.txt', 'a') as outf:
        generate_sample(1024, 500, 0.3, outf=outf)
        # walk_samples(True, outf)
