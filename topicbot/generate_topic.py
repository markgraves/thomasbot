import argparse
import os, sys
import subprocess
from subprocess import call, check_output

def generate_sample(vsize=512, epochs=50, words=10000, temperature=1, ttopic=-1, gtopic=-1, cuda=False, outf=None):
    pp_ttopic = ttopic if ttopic > 0 else 'all' if ttopic == 0 else ''
    pp_gtopic = gtopic if gtopic > 0 else 'all' if gtopic == 0 else ''
    out_file = 'generated/generate_aquinas_summa_topic_pt2_wfull_bpe_%s_beos_l2_s256_emb256' % vsize
    if pp_ttopic:
        out_file += '_topic%s' % pp_ttopic
    out_file += '_e%s' % epochs
    if temperature > 0:
        out_file += '_temp%s' % str(temperature).replace('.', 'd')
    out_file += '_w%s' % words
    out_file += '_top%s'% pp_gtopic
    gen_cmd = 'python generate.py --data data/aquinas_summa_topic_pt2_wfull_bpe_%s_beos --checkpoint results/aquinas_summa_topic_pt2_wfull_bpe_%s_beos_l2_s256_emb256_topic%s_e%s.pt --outf %s-raw.txt --words %s' % (vsize, vsize, pp_ttopic, epochs, out_file, words)
    if temperature:
        gen_cmd += ' --temperature %s' % temperature
    gen_cmd += ' --topicnum %s' % gtopic
    if cuda:
        gen_cmd += ' --cuda'
    decode_cmd = 'spm_decode --extra_options=bos:eos --input_format piece --model=$HOME/work/aquinas-sentencepiece/models/aquinas_summa_full_%s_bpe.model --output %s.txt %s-raw.txt' % (vsize, out_file, out_file)
    print(gen_cmd)
    call(gen_cmd, shell=True)
    print(decode_cmd)
    call(decode_cmd, shell=True)
    length = int(check_output('wc -w ' + out_file + '.txt', shell=True).split()[0])
    misspelled = int(check_output('enchant -l %s.txt|grep -v "eos"|wc -l' % out_file, shell=True).strip())
    misspelled_freq = float(misspelled) / float(length)
    print(out_file + '.txt')
    print('words %s; misspelled %s; freq misspelled %s' % (length, misspelled, misspelled_freq))
    if outf:
        outf.write('\t'.join(str(x) for x in [vsize, pp_ttopic, pp_gtopic, words, temperature, length, misspelled, misspelled_freq, out_file]) + '\n')

def walk_samples(cuda=False, outf=None):
    for temperature in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for vsize in ['256', '512', '1k']:
            generate_sample(vsize, 50000, temperature, cuda=cuda, outf=outf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--ttopic', type=int, default=-1,
                        help='train topic number (default -1 for none, 0 for all)')
    parser.add_argument('--gtopic', type=int, default=-1,
                        help='generate topic number (default -1 for none, 0 for all)')
    parser.add_argument('--vsize', type=int, default=512,
                        help='size of vocabulary')
    parser.add_argument('--emsize', type=int, default=256,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--words', type=int, default='1000',
                        help='number of words to generate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    args = parser.parse_args()
    generate_sample(vsize=args.vsize, epochs=args.epochs, words=args.words, temperature=args.temperature, gtopic=args.gtopic, ttopic=args.ttopic, cuda=args.cuda)
