#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
    
def split3(infile, classfile=None):
    questions = []
    qtext = ''
    with open(infile, 'r') as finput:
        if classfile:
            cinput = open(classfile, 'r')
            with open('header.txt', 'w') as hfile:
                hfile.write('article'+cinput.readline().strip()+'\n')  #article colhead is missing from csv
                print('Wrote header.txt')
        for line in finput.readlines():
            line_strip = line.strip()
            if line_strip in ['QUESTION', '▁Q U ESTION', '<s> ▁Q U ESTION </s>', '<s> ▁Q U E S TI O N </s>', 
                              '<s> ▁ Q U E S T I O N </s>'] or line_strip.startswith('<question>'):
                if qtext:
                    # have question text
                    questions.append(qtext)
                else:
                    pass  # first time through
                qtext = line  # start next question with the "QUESTION" text
            elif line_strip.startswith('<s> ▁A R TI C L E') or line_strip.startswith('<article>'):  # only works w 512 vocab size or tagged
                if classfile:
                    qtext += '<data csv="%s"/>\n' % cinput.readline().strip()
                qtext += line
            else:
                # build up qtext
                qtext += line
        #save last question, if any text
        if qtext:
            questions.append(qtext)
        if classfile:
            cinput.close()
            
    train, valid_test = train_test_split(questions, test_size=.25, random_state=20180103)
    valid, test = train_test_split(valid_test, test_size=.4, random_state=20180103)    # .15 valid, .10 test
    print('Splitting %s questions: %s %s %s' % (len(questions), len(train), len(valid), len(test)))
    print('Recommended test: cat train.txt test.txt valid.txt |wc; wc %s' % infile)
    
    with open('train.txt', 'w') as ftrain:
        for question in train:
            ftrain.write(question)
    with open('test.txt', 'w') as ftest:
        for question in test:
            ftest.write(question)
    with open('valid.txt', 'w') as fvalid:
        for question in valid:
            fvalid.write(question)
            

if __name__ == '__main__':
    if len(sys.argv) == 2:
        split3(sys.argv[1])
    elif len(sys.argv) == 3:
        split3(sys.argv[1], sys.argv[2])
