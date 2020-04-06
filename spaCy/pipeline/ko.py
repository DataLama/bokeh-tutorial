import os
import re
import string
import pickle
from typing import List, Tuple,Dict
from soynlp.tokenizer import LTokenizer
from soynlp.normalizer import repeat_normalize
from emoji import emojize, demojize

from spacy.tokens import Doc, Token
from spacy.lang.ko import Korean
from spacy.util import DummyTokenizer

class Tokenizer:
    """Tokenizer class"""
    def __init__(
        self
    ):
        # load noun cohesion score
        with open('pipeline/words.p', 'rb') as rf:
            words = pickle.load(rf)    
            cohesion_score = {word: score.cohesion_forward for word, score in words.items()}
            cohesion_score = {k: v for k, v in sorted(cohesion_score.items(), key=lambda item: item[1], reverse=True) if v > 0}      
        with open('pipeline/nouns.p', 'rb') as rf:
            nouns = pickle.load(rf)
            noun_score = {noun: score.score for noun, score in nouns.items()}
            noun_cohesion_score = {noun: score + cohesion_score.get(noun, 0) for noun, score in noun_score.items()} 
            self._noun_cohesion_score = {k: v for k, v in sorted(noun_cohesion_score.items(), key=lambda item: item[1], reverse=True) if v > 0}
        
        self._soy = LTokenizer(scores=self._noun_cohesion_score)
        self._is_flatten = False # no_flatten
        self._is_remove_r = False # no_remove
        
    def _preprocess(self, doc: str) -> str:
        """전처리 로직"""
        doc = str(doc).lower().strip() # make string, lower and strip
#         doc = re.sub(r'[\d+‘’]','', doc) # 연속하는 숫자 제거
#         doc = re.sub(f'[{string.punctuation}]','', doc) # 기초 특수문자 제거
        doc = repeat_normalize(doc, num_repeats=3) # 연속 3글자 이상의 글자 normalize하기.
        doc = emojize(demojize(doc, delimiters=(' :', ': '))).strip() # 이모지 사이에 공백 추가.
        return doc
    
    def _postprocess(self, doc:List[Tuple[str]]) -> List[Dict]:
        """후처리 로직"""
        processed_doc = []
        for l_part, r_part in doc:
            processed_doc.append({'words': l_part,'spaces':False ,'tag_' : 'L'})        
            if r_part !='':
                processed_doc.append({'words': r_part ,'spaces':True ,'tag_' : 'R'})
            else:
                processed_doc[-1].update({'spaces':True})

        if processed_doc != []:
            processed_doc[-1].update({'spaces':False}) # 마지막은 무조건 false
        return processed_doc
        
    def tokenize(self, doc: str, media_type: str = None) -> List[str]:
        """tokenize function
        Use noun cohesion score with soynlp
        
        doc :
        media_type :
        """
        
        doc = self._soy.tokenize(self._preprocess(doc), flatten=self._is_flatten, remove_r = self._is_remove_r) # returns list of tuple
        doc = self._postprocess(doc) 
        
        return doc

    
class SpacyTokenizer(DummyTokenizer): # wrapping for spacy
    def __init__(self, vocab):
        self._vocab = vocab # 해당언어에 대한...
        self._tokenizer = Tokenizer()        
    
    def __call__(self, text):
        tokenized_list = self._tokenizer.tokenize(text)
        words, spaces, tag_ = list(zip(*[(token['words'], token['spaces'], token['tag_']) 
                                                     for token in tokenized_list]))
        
        doc = Doc(self._vocab, words=words, spaces=spaces)
        
        for token, t in zip(doc, tag_):
            token._.set('tag_', t)
        
        return doc    
    
    
if __name__ == "__main__":
    #### Tokenizer Examples
    import ray
    ray.init(num_cpus=64, object_store_memory = 200000 * 1024 * 1024, driver_object_store_memory = 100000 * 1024 * 1024)
    import modin.pandas as pd
    
    ## load data and define variables
    file_path = 'data/Korean.parquet'
    cols = ['Id', 'Media Type', 'Mention Content']
    df = pd.read_parquet(file_path, columns=cols)
    
    ## tokenize with modin apply
    tokenized_content = (
                     df['Mention Content']
                        .apply(lambda x:tokenizer_ko.tokenize(x) if x != None else x, axis=1)   
                    )
    print(len(tokenized_content))
    
    ray.shutdown()