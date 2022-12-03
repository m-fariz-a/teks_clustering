from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from translate import Translator

class StringOperation:
    
    def remove_nonalphabet(series_text):
        """
        Remove non alphabet from a text series (number and other character)

        parameter
        ----------
        seriest_text : Series
            text input with series dtype

        return
        ------
        series_text_output : Series
            output result
        """

        series_text_output = series_text.copy()
        series_text_output = series_text_output.str.lower()

        # remove non alphabet character
        series_text_output = series_text_output.str.replace(r'[^a-z ]+', '', regex=True)

        return series_text_output
    
    
    
    def remove_stopwords(series_text, stopword_list):
        """
        remove stopword from text
        
        parameter
        ----------
        seriest_text : Series
            text input with series dtype
        stopword_list : list
            list of stopwords

        return
        ------
        series_text_output : Series
            output result
        """
        
        series_text_output = series_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopword_list)]))
        
        return series_text_output
    
    
    
    def word_stemming(series_text):
        """
        get a standard word
        
        parameter
        ----------
        seriest_text : Series
            text input with series dtype

        return
        ------
        series_text_output : Series
            output result
        """
        
        stemmer = PorterStemmer()
        tokenizer = word_tokenize
        
        series_text_output = series_text.apply(lambda x: ' '.join([stemmer.stem(word) for word in tokenizer(x)]))
        
        return series_text_output
    
    
    
    def word_translation(series_text,
                        src_lang='id', dest_lang='en'):    
        """
        get translate into specific language
        
        parameter
        ----------
        seriest_text : Series
            text input with series dtype
        src_lang : text
            source language code
        dest_lang : text
            destination language code

        return
        ------
        series_text_output : Series
            output result
        """
        
        translator= Translator(from_lang=src_lang, to_lang=dest_lang)
        
        series_text_output = series_text.apply(lambda x: translator.translate(x))
        
        return series_text_output