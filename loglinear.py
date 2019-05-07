import nltk
import operator
import functools
from nltk.corpus import brown



#class LogLinearTagger(nltk.TaggerI):
class LogLinearTagger:

    def __init__(self,training_corpus):
        self.classifier = None
        self.training_corpus = training_corpus

    def train(self):
        self.classifier = nltk.MaxentClassifier.train(
                    functools.reduce(operator.add,
                        list(map(lambda tagged_sent :
                            self.sent_to_feature(tagged_sent)
                            ,self.training_corpus)),algorithm='megam'))

    def sent_to_feature(self,tagged_sent):
        return  map(lambda i, elem :
                        apply( lambda token , tag :
                             (self.extract_features(token, i, tag), elem[1]),zip(*tagged_sent)),enumerate(tagged_sent))

    def tag_sentence(self, sentence_tag):
        if self.classifier == None:
            self.train()
        return apply (lambda sentence :
                    zip(sentence,
                    functools.reduce(lambda x,y:
                        apply(operator.add,
                            [x,[self.classifier.classify(self.extract_features(sentence, y[0], x))]])
                        , enumerate(sentence), []))
                    ,[map(operator.itemgetter(0),sentence_tag)])

    def evaluate(self,test_sents):
        return apply(lambda result_list :
                    sum(result_list)/float(len(result_list))
                    , [functools.reduce(operator.add,
                        map(lambda line:
                            map(lambda tag : int(tag[0] == tag[1])
                                , zip(map(operator.itemgetter(1),line),
                                      map(operator.itemgetter(1),self.tag_sentence(line))))
                           ,test_sents))])

    def extract_features(self, sentence, i, history):
        features = {}
        features["this-word"] =  sentence[i]
        if i == 0:
            features["prev-tag"] = "<START>"
        else:
            features["prev-tag"] = history[i-1]
        return features





