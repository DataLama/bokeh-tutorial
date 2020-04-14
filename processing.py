import os
import re
import emoji
import string
import pickle
import json
import pandas as pd
import argparse
from itertools import chain
from collections import Counter
from utils.vocab import Vocab
from multiprocessing import Pool
from datetime import datetime
from pipeline import ko

"""
우선 한국어만 제대로 만들고 scale out하자.
"""

TEXT_COLS = ["Mention Title", "Mention Content"]
FEATURE_COLS =["Id", "Date", "Country", "State", "City", "Media Type", "Mention URL", "Publisher Name", "Publisher Username", "Site Domain", "Site Name", "Topics", "Product_Category"] 

class Transform(object):
    def __init__(self, vocab_path, stopwords_path, tokenizer):
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(stopwords_path) as f:
            self.STOPWORDS = [s.strip() for s in f.readlines()]
        self.emo = emoji.get_emoji_regexp()
        self.tokenizer = tokenizer
        
    def _get_keywords(self, token_list, pos):
        """
        step 1) stopwords 제외하기.
        step 2) pos == L part일 경우, Punctuation, 이모지, 해시태그 제외한 모든 것.
        """
        keywords = []
        for token, pos in zip(token_list, pos):
            token_ind = self.vocab.to_indices(token)
            if token in self.STOPWORDS:
                continue
            elif pos == 'L':
                if not self.vocab.lexeme['is_Punct'][token_ind] and not self.vocab.lexeme['is_Emoji'][token_ind]:
                    keywords.append(token)
            else:
                continue
                
        return keywords
        
    def _gen_ngrams(self, doc, n_gram=1):
        ngrams = zip(*[doc[i:] for i in range(n_gram)])
        return [" ".join(ngram) for ngram in ngrams]
        
    
    def transform(self, doc):
        """
        doc = (doc, {"Id": id_value, ...})
        """
        document = doc[0]
        tokenized = list(zip(*self.tokenizer.tokenize(document)))
        if tokenized == []:
            token_list = []
            pos = []
        else:
            token_list = list(tokenized[0])
            pos = list(tokenized[1])
        
        document = re.sub(r'(http\S+[^가-힣])|([a-zA-Z]+.\S+.\S+[^가-힣])',r' ', document) # url
        document = re.sub(r'(\[image#0\d\])', r' ', document) # image
        document = re.sub(r'([0-9a-zA-Z_]|[^\s^\w])+(@)[a-zA-Z]+.[a-zA-Z)]+',r' ', document) # email
        
        ## keyword_list
        keywords = self._get_keywords(token_list, pos) # 쓸데 없는 태그를 다 제거하고 순수 keyword만 추출
        bigram = self._gen_ngrams(keywords, n_gram=2)
        trigram = self._gen_ngrams(keywords, n_gram=3)
                
        ProcessedDoc = {
            "Id":doc[1].pop('Id'),
            "Feature": doc[1],
            "Document":{
                "Mention": document
            },
            "Token":{
                "Tokenized_list": token_list,
                "Pos": pos,
                "KEYWORDS_UNI": keywords, # keywords unigram !=  token_list
                "KEYWORDS_BI": bigram,
                "KEYWORDS_TRI": trigram,
                "HASHTAGS": re.findall(r'#(\w+)', document),
                "EMOJI": self.emo.findall(document),
            }
        }
        return ProcessedDoc


def main(args):
    print("Load Tokenizer and Define Variables.")
    ## by arguments
    if args.lang == 'ko':
        tokenizer = ko.Tokenizer()
    else:
        raise ValueError("Wrong arguments for --lang. Please pass 'ko' for --lang arguments.")

    processed_path = args.path 
    vocab_path = f'{processed_path}/vocab-20200413.pkl'
    stopwords_path = 'utils/STOPWORDS_KO.txt'
    transform = Transform(vocab_path=vocab_path, stopwords_path=stopwords_path, tokenizer=tokenizer)
    
    ## load data
    cols = TEXT_COLS + FEATURE_COLS
    df = pd.read_parquet('data/Korean.parquet', columns=cols)
    category = pd.read_csv('/data/social_buzz_dataset/ko-tag-classification.csv', usecols = ['Id','Product_Category'])
    df = pd.merge(category, df, how='left')
    
    mention = df['Mention Title'] + ' ' + df['Mention Content']
    mention = mention.fillna('')
    features = df.loc[:,FEATURE_COLS]
    
    docs = [(doc, context) for doc, context in zip(mention, features.to_dict('r'))] # (str, {'Id':,,,,'Media Type':,,,,})
    
    #### Transform_data
    with Pool(processes=os.cpu_count()) as pool:
        processed_docs = pool.map(transform.transform, docs)
        
    os.makedirs(f'{processed_path}/documents', exist_ok=True)
    for doc in processed_docs:
        with open(f'{processed_path}/documents/{doc["Id"]}.json', 'w') as f:
            json.dump(doc,f)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline.")
    parser.add_argument('--lang', default='ko', help="Which language vocab you?")
    parser.add_argument('--path', default='data/ProcessedText', help="Path where the vocab is saved.")
    args = parser.parse_args()
    main(args)