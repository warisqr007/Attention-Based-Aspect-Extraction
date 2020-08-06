from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs
import json
from tqdm import tqdm
import argparse
import pandas as pd


def parseSentence(line):
    lmtzr = WordNetLemmatizer()
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    parsedText = ' '.join(text_stem)
    return parsedText


def preprocess_train(domain):
    #f = codecs.open('../datasets/' + domain + '/train.txt', 'r', 'utf-8')
    #out = codecs.open('../preprocessed_data/' + domain + '/train.txt', 'w', 'utf-8')
    
    in_path = '../datasets/%s/train.pkl' % (domain)
    out_path = '../preprocessed_data/%s/train.pkl' % (domain)
    
    data = pd.read_pickle(in_path)
    data['content'] = data.apply(lambda row : parseSentence(row['content']), axis=1)
    
    data.to_pickle(out_path)

    '''
    for line in f:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            out.write(' '.join(tokens) + '\n')
    '''


def preprocess_test(domain):
    # For restaurant domain, only keep sentences with single 
    # aspect label that in {Food, Staff, Ambience}

    f1 = codecs.open('../datasets/' + domain + '/test.txt', 'r', 'utf-8')
    f2 = codecs.open('../datasets/' + domain + '/test_label.txt', 'r', 'utf-8')
    out1 = codecs.open('../preprocessed_data/' + domain + '/test.txt', 'w', 'utf-8')
    out2 = codecs.open('../preprocessed_data/' + domain + '/test_label.txt', 'w', 'utf-8')

    for text, label in zip(f1, f2):
        label = label.strip()
        if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
            continue
        tokens = parseSentence(text)
        if len(tokens) > 0:
            out1.write(' '.join(tokens) + '\n')
            out2.write(label + '\n')


def preprocess(domain, require_test=True):
    print('\t' + domain + ' train set ...')
    preprocess_train(domain)
    if(require_test):
        print('\t' + domain + ' test set ...')
        preprocess_test(domain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant',
                        help="domain of the corpus")
    parser.add_argument("--require_test", dest="require_test", action='store_true', default=False,
                        help="require to run on test set")
    args = parser.parse_args()

    preprocess(args.domain, args.require_test)
