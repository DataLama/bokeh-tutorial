import os
import re
import string
import pickle
from typing import List, Tuple
from soynlp.tokenizer import LTokenizer
from soynlp.normalizer import repeat_normalize
from emoji import emojize, demojize, get_emoji_regexp
from spacy.tokens import Doc, Token
from spacy.lang.ko import Korean
from spacy.util import DummyTokenizer

class Tokenizer:
    """Tokenizer class"""
    def __init__(
        self
    ):
        # load noun cohesion score
        with open('utils/words.p', 'rb') as rf:
            words = pickle.load(rf)    
            cohesion_score = {word: score.cohesion_forward for word, score in words.items()}
            cohesion_score = {k: v for k, v in sorted(cohesion_score.items(), key=lambda item: item[1], reverse=True) if v > 0}      
        with open('utils/nouns.p', 'rb') as rf:
            nouns = pickle.load(rf)
            noun_score = {noun: score.score for noun, score in nouns.items()}
            noun_cohesion_score = {noun: score + cohesion_score.get(noun, 0) for noun, score in noun_score.items()} 
            self._noun_cohesion_score = {k: v for k, v in sorted(noun_cohesion_score.items(), key=lambda item: item[1], reverse=True) if v > 0}
        
        self._soy = LTokenizer(scores=self._noun_cohesion_score)
        self._is_flatten = False # no_flatten
        self._is_remove_r = False # no_remove
        self._emo = get_emoji_regexp() # re compiled
        
        
    def _preprocess(self, doc: str) -> str:
        """전처리 로직"""
        doc = str(doc).lower().strip() # make string, lower and strip
        doc = re.sub(rf'([^{self._emo.pattern}{string.punctuation}\s\w]+)',' ', doc) # 숫자, 문자, whitespace, 이모지, 일반특수문자를 제외한 모든 유니코드 제거.
        doc = re.sub(r'\s',' ', doc) #white space character 변환 
        doc = re.sub('&nbsp;',' ', doc) #&nbsp; 제거
        doc = re.sub('&lt;','<',doc) #기타 html특수기호
        doc = re.sub('&gt;','>',doc) #기타 html특수기호
        doc = re.sub('&amp;','&',doc) #기타 html특수기호
        doc = re.sub('&quot;','""',doc) #기타 html특수기호
        doc = re.sub(r'(http\S+[^가-힣])|([a-zA-Z]+.\S+.\S+[^가-힣])',r' [URL] ',doc) #url 변환
        doc = re.sub(r'(\[image#0\d\])', r' [IMAGE] ', doc) # Image Tag
        doc = re.sub(r'([0-9a-zA-Z_]|[^\s^\w])+(@)[a-zA-Z]+.[a-zA-Z)]+',r' [EMAIL] ', doc) #email
        doc = re.sub(r'#(\w+)',r' [HASHTAG] ', doc) #Hashtag
        doc = re.sub(r'@(\w+)',r' [MENTION] ', doc) #MENTION
        doc = emojize(demojize(doc, delimiters=(' :', ': '))).strip()
        return doc
    
    def _postprocess(self, doc:List[str]) -> List[Tuple[str]]:
        """후처리 로직"""
        processed_doc = []
        
        for l_part, r_part in doc:
            
            ## l_part
            l_part = repeat_normalize(l_part, num_repeats=3)
            sub_l_part = re.findall(r"[\w]+|[\W]+", l_part)
            if len(sub_l_part)==2:
                processed_doc += [(sub, 'L') for sub in sub_l_part] 
            else:
                processed_doc.append((l_part, 'L'))      
            
            ## r_part
            if r_part !='':
                r_part = repeat_normalize(r_part, num_repeats=3)
                sub_r_part = re.findall(r"[\w]+|[\W]+", r_part)
                if len(sub_r_part)==2:
                    processed_doc += [(sub, 'R') for sub in sub_r_part] 
                else:
                    processed_doc.append((r_part, 'R'))
            
                
        return processed_doc
        
    def tokenize(self, doc: str, media_type: str = None) -> List[Tuple[str]]:
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
