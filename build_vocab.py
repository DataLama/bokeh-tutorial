import os
import re
import emoji
import string
import pickle
import json
import argparse
import pandas as pd
from itertools import chain
from collections import Counter
from utils.vocab import Vocab
from multiprocessing import Pool
from datetime import datetime
from pipeline import ko

"""
우선 한국어만 제대로 만들고 scale out하자.
"""

tokenizer = ko.Tokenizer

def main(args):
    print("Load Tokenizer and Define Variables.")
    ## by arguments
    if args.lang == 'ko':
        tokenizer = ko.Tokenizer()
    else:
        raise ValueError("Wrong arguments for --lang. Please pass 'ko' for --lang arguments.")
    processed_path = args.path
    
    ## etc
    emo = emoji.get_emoji_regexp()
    now = datetime.now()
    
    ## Load data for synthesio
    cols = ['Mention Title','Mention Content']
    df = pd.read_parquet('data/Korean.parquet', columns=cols)
    df = df.fillna('')
    docs = [doc for doc in df['Mention Title'] + ' ' + df['Mention Content']] 
    
    print("Tokenize the documents and build the vocab.")
    with Pool(processes=os.cpu_count()) as pool:
        tokenized_docs = pool.map(tokenizer.tokenize, docs)
        
    token_counts = Counter(list(zip(*chain(*tokenized_docs)))[0]).most_common()
    vocab = Vocab(list_of_tokens=[token for token, count in token_counts if count >= int(args.min_count)], token_to_idx={'[PAD]':0,'[UNK]':1})
    vocab.lexeme['is_Emoji'] = [True if emo.fullmatch(term) !=None else False for term in vocab.idx_to_token]
    vocab.lexeme['is_Digit'] = [True if re.fullmatch(r'[\d\,\.]+',term) !=None else False for term in vocab.idx_to_token]
    vocab.lexeme['is_Punct'] = [True if re.fullmatch(rf'[{string.punctuation}]+',term) !=None else False for term in vocab.idx_to_token]

    print(f"Build the new vocab vocab-size : {len(vocab)}")
    with open(f"{processed_path}/vocab-{now:%Y%m%d}.pkl", 'wb') as f:
        pickle.dump(vocab, f)
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Build Vocab")
    parser.add_argument('--lang', default='ko', help="Which language vocab you?")
    parser.add_argument('--path', default='data/ProcessedText', help="Path where the vocab is saved.")
    parser.add_argument('--min_count', default='5', help="Minimun count for vocab") # min_value 5 이상...
    args = parser.parse_args()
    main(args)