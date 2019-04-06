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
            if line.strip() in ['QUESTION', '▁Q U ESTION', '<s> ▁Q U ESTION </s>', '<s> ▁Q U E S TI O N </s>', '<s> ▁ Q U E S T I O N </s>']:
                if qtext:
                    # have question text
                    questions.append(qtext)
                else:
                    pass  # first time through
                qtext = line  # start next question with the "QUESTION" text
            elif line.strip().startswith('<s> ▁A R TI C L E'):  # only works w 512 vocab size
                if classfile:
                    qtext += '<data csv="%s"/>\n' % cinput.readline().strip()
                qtext += line
            else:
                # build up qtext
                qtext += line
        #save last question, if any text
        if qtext:
            questions.append(qtext)
        cinput.close()
            
    train, valid_test = train_test_split(questions, test_size=.2, random_state=20180103)
    valid, test = train_test_split(valid_test, test_size=.5, random_state=20180103)    # .10 valid, .10 test
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
