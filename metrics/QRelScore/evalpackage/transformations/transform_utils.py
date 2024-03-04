import os
import re
import copy
import string
import json
import random
random.seed(32767)
import numpy as np
np.random.seed(32767)
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def word_untokenize(tokens):
    '''
        I've -> I 've -> I've
        don't -> do n't -> don't
        can't -> ca n't -> can't
        cannot -> can not -> can not
    '''
    return ''.join([' ' + i if not i.startswith("'") and not i.startswith("n'") and i not in string.punctuation else i for i in tokens]).lstrip()


# python built-in type system
def align_ws(source_token, target_token):
    # Align trailing whitespaces between tokens
    if target_token[-1] == source_token[-1] == ' ':
        return source_token
    elif target_token[-1] == ' ':
        return source_token + ' '
    elif source_token[-1] == ' ':
        return source_token[:-1]
    else:
        return source_token


# python built-in type system
def align_case(source_token, target_token):
    return source_token.lower() if target_token.islower() else source_token.capitalize()


class ParaphrasingMixIn:
    '''
        Prerequisite object/instance-related attributes: self.data_dict, self.sentences_dict, self.forward_translate_model, self.backward_translate_model,
                                                         self.forward_translate_tokenizer, self.backward_translate_tokenizer
    '''
    def paraphrasing_transformation(self):
        keys_list, sentences_list = list(self.sentences_dict.keys()), list(self.sentences_dict.values())

        # forward translate en -> others
        intermediate_language_codes = self.forward_translate_tokenizer.supported_language_codes
        intermediate_tokens = self.translate(self.forward_translate_model, self.forward_translate_tokenizer, sentences_list, '>>fra<<')

        # backward translate others -> en
        round_trip_tokens = self.translate(self.backward_translate_model, self.backward_translate_tokenizer, intermediate_tokens)

        # ranking based on round trip probablity
        pass

        paraphrased_dict = dict(zip(keys_list, round_trip_tokens))

        return paraphrased_dict

    def translate(self, model, tokenizer, sentences_list, target_language_code = None):
        template = lambda sentence: ('{} {}'.format(target_language_code, sentence) if target_language_code is not None else sentence)

        wrapped_list = [ template(s) for s in sentences_list ]
        model_input = tokenizer(wrapped_list, return_tensors = 'pt', padding = True) # wrap all samples into a batch with aligned length

        for key in model_input.keys():
            model_input[key] = torch.LongTensor(model_input[key]).cuda(0)

        translated_token_ids = model.generate(**model_input)

        translated_tokens = tokenizer.batch_decode(translated_token_ids, skip_special_tokens=True)

        return translated_tokens


class NegationMixIn:
    '''
        Prerequisite object/instance-related attributes: self.data_dict, self.sentences_list, self.available_entities, self.negation_words, self.modal_verbs
    '''
    def negation_transformation(self):
        keys_list, sentences_list = list(self.sentences_dict.keys()), list(self.sentences_dict.values())

        negated_sentences_list = [  self.negation_sentence_level(s) for s in sentences_list ]

        negated_dict = dict(zip(keys_list, negated_sentences_list))

        return negated_dict

    def negation_sentence_level(self, sentence):
        verb_nagations = [ 'not', ] #"n't" ]
        tokens = word_tokenize(sentence)
        negated = False
        for i, token in enumerate(tokens):
            if token in self.negation_words:
                del tokens[i]
                negated = True
                break
            elif token in self.modal_verbs:
                # modal verbs, be-verbs and auxiliary verbs
                tokens.insert(i + 1, 'not')
                negated = True
                break

        if not negated:
            pos_sen = pos_tag(tokens)
            for i, (token, token_tag) in enumerate(pos_sen):
                if token_tag in { 'VB', 'VBP' }: # verb base form, predicate verb non-3rd person singular present
                    tokens.insert(i, "don't")
                    break
                elif token_tag == "VBD": # verb past tense
                    tokens[i] = lemmatizer.lemmatize(token, 'v')
                    tokens.insert(i, "didn't")
                    break
                elif token_tag in { "VBG", 'VBN' }: # verb gerund or present participle, verb past participle
                    tokens.insert(i, 'not')
                    break
                elif token_tag == "VBZ": # verb 3rd person singular present
                    tokens[i] = lemmatizer.lemmatize(token, 'v')
                    tokens.insert(i, "doesn't")
                    break

        return word_untokenize(tokens)


class NERSwapMixIn:
    '''
        Base class for swap-based mix-in transformations: such as ConceptSwap, NumberSwap, DateSwap ...
        Prerequisite object/instance-related attributes: self.data_dict, self.sentences_list, self.spacy
    '''
    def nerswap_transformation(self, categories):
        keys_list = [ ]
        nerswapped_sentences_list = [ ]

        for qid, d in self.data_dict.items():
            keys_list.append(qid)
            sentence = d['gold_question']
            context = d['context']
            nerswapped_sentence = self.nerswap_sentence_level(sentence, context, categories)
            nerswapped_sentences_list.append(nerswapped_sentence)

        nerswapped_dict = dict(zip(keys_list, nerswapped_sentences_list))

        return nerswapped_dict
    
    def nerswap_sentence_level(self, sentence, context, categories):
        spacy_sentence, spacy_context = self.spacy(sentence), self.spacy(context)

        # python built-in type system
        original_sentence_tokens = [ token.text_with_ws for token in spacy_sentence ]
        swapped_sentence_tokens = original_sentence_tokens # In case swapping operation fails

        # spacy type system
        available_sentence_ents = list(filter(lambda e: e.label_ in categories, spacy_sentence.ents))
        available_context_ents = list(filter(lambda e: e.label_ in categories, spacy_context.ents))

        if available_sentence_ents:
            target_entity = random.choice(available_sentence_ents)
            target_text = target_entity.text
            filter_in_condition = lambda e: e.text != target_text and e.text not in target_text and target_text not in e.text
            candidate_source_entities = list(filter(filter_in_condition, available_context_ents))

            if candidate_source_entities:
                source_entity = random.choice(candidate_source_entities)

                # python built-in type system
                aligned_source_tokens_in_a_single_str = align_ws(source_entity.text, target_entity.text_with_ws)
                s = target_entity_start_token_index_in_spacy_sentence = target_entity.start
                e = target_entity_end_token_index_in_spacy_sentence = target_entity.end
                # list of strs: token + a str consisting of several tokens
                swapped_sentence_tokens = original_sentence_tokens[:s] + [ aligned_source_tokens_in_a_single_str ] + original_sentence_tokens[e:]

        return ''.join(swapped_sentence_tokens)


class ConceptSwapMixIn(NERSwapMixIn):
    def conceptswap_transformation(self):
        categories = [ 'PERSON', 'ORG', 'NORP', 'FAC', 'GPE', 'LOC', 'PRODUCT', 'WORK_OF_ART', 'EVENT' ]
        conceptswapped_dict = super().nerswap_transformation(categories)

        return conceptswapped_dict


class NumberSwapMixIn(NERSwapMixIn):
    def numberswap_transformation(self):
        categories = [ 'PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL' ]
        numberswapped_dict = super().nerswap_transformation(categories)

        return numberswapped_dict


class DateTimeSwapMixIn(NERSwapMixIn):
    def datetimeswap_transformation(self):
        categories = [ 'DATE', 'TIME' ]
        datetimeswapped_dict = super().nerswap_transformation(categories)

        return datetimeswapped_dict    


class PronounSwapMixIn:
    '''
        Prerequisite object/instance-related attributes: self.class2pronoun, self.pronoun2class, self.pronouns, self.spacy
    '''
    def pronounswap_transformation(self):
        keys_list = [ ]
        pronounswapped_sentences_list = [ ]

        for qid, d in self.data_dict.items():
            keys_list.append(qid)
            sentence = d['gold_question']
            context = d['context']
            pronounswapped_sentence = self.pronounswap_sentence_level(sentence)
            pronounswapped_sentences_list.append(pronounswapped_sentence)

        pronounswapped_dict = dict(zip(keys_list, pronounswapped_sentences_list))

        return pronounswapped_dict
    
    def pronounswap_sentence_level(self, sentence):
        spacy_sentence = self.spacy(sentence)

        sentence_pronouns = [ token for token in spacy_sentence if token.text.lower() in self.pronouns ]

        original_sentence_tokens = [ token.text_with_ws for token in spacy_sentence ]
        swapped_sentence_tokens = original_sentence_tokens # In case swapping operation fails

        if sentence_pronouns:
            target_pronoun = random.choice(sentence_pronouns)
            # pronouns are always in a single token, which makes the swapping operation much easier than entity-based ones
            target_token_index = target_pronoun.i
            target_pronoun_uncased_str = target_pronoun.text.lower()

            if target_pronoun_uncased_str in self.pronoun2class:
                target_pronoun_class = self.pronoun2class[target_pronoun_uncased_str]
                candidate_pronouns = self.class2pronoun[target_pronoun_class].copy()
                candidate_pronouns.remove(target_pronoun_uncased_str)

                source_pronoun_uncased_str = random.choice(candidate_pronouns)
                aligned_source_pronoun_str = align_ws(source_pronoun_uncased_str, target_pronoun.text_with_ws)
                aligned_source_pronoun_str = align_case(aligned_source_pronoun_str, target_pronoun.text)

                swapped_sentence_tokens[target_token_index] = aligned_source_pronoun_str
        
        return ''.join(swapped_sentence_tokens)


class NoiseInjectionMixIn:
    pass


class SQUADTransformer(ParaphrasingMixIn, NegationMixIn, ConceptSwapMixIn, NumberSwapMixIn, DateTimeSwapMixIn, PronounSwapMixIn, NoiseInjectionMixIn):
    '''
        Transform the dataset (with a specific format) into a relatively flattened and unified structure with
        varities of elaborated corrupting operations such as paraphrasing, swapping and negation.
    '''
    def __init__(self, dataset_filename, target_dirname):
        self.dataset_filename = dataset_filename
        self.target_dirname = target_dirname
        self.parse_input()
        self.build_model()
        self.build_entities_and_negation_words()
        self.build_modal_verbs()
        self.build_pronouns()

    def parse_input(self):
        with open(self.dataset_filename) as f:
            train_set = json.load(f)
        data_file = train_set

        self.data_dict = { }
        self.sentences_dict = { }

        for doc in data_file['data']:
            title = doc['title']
            for par in doc['paragraphs']:
                context = par['context']
                for qa in par['qas']:
                    qid = qa['id']
                    self.data_dict[qid] = {
                        'context': context,
                        'answers': qa['answers'], # multiple answers for a given question
                        'gold_question': qa['question'],
                        'corrupted_question': { }
                    }
                    self.sentences_dict[qid] = qa['question']

    def build_model(self):
        # back-translation based paraphrasing
        forward_model_name = 'Helsinki-NLP/opus-mt-en-roa'
        self.forward_translate_model = MarianMTModel.from_pretrained(forward_model_name).cuda(0)
        self.forward_translate_tokenizer = MarianTokenizer.from_pretrained(forward_model_name)

        backward_model_name = 'Helsinki-NLP/opus-mt-roa-en'
        self.backward_translate_model = MarianMTModel.from_pretrained(backward_model_name).cuda(0)
        self.backward_translate_tokenizer = MarianTokenizer.from_pretrained(backward_model_name)

        self.spacy = spacy.load('en_core_web_sm')

    def build_entities_and_negation_words(self):
        # build available entities
        concept_entity_filename = '/path/to/home/to/your/project/qgbase/evalpackage/transformations/conceptnet_entity.csv'
        self.available_entities = set()

        with open(concept_entity_filename) as f:
            for line in f:
                word = ' '.join(line.strip().split('|||')[:-1])
                self.available_entities.add(word) # `join()` is used to port the entity with two or more tokens
        
        sw = set(stopwords.words('english'))
        self.available_entities -= sw

        # build negation words
        negation_filename = '/path/to/home/to/your/project/qgbase/evalpackage/transformations/negation.txt'
        self.negation_words = [ ]

        with open(negation_filename) as f:
            for line in f:
                word = ' '.join(line.strip().split()[1:]) # `join()` is used to port the negation with two or more tokens
                self.negation_words.append(word)
                self.available_entities.add(word)

        # build extra words
        extra_words = [ '.', ',', '!', '?', 'male', 'female', 'neutral' ]
        for ew in extra_words:
            self.available_entities.add(ew)

    def build_modal_verbs(self):
        modal_verbs_filename = '/path/to/home/to/your/project/qgbase/evalpackage/transformations/modal.txt'
        self.modal_verbs = set()

        with open(modal_verbs_filename) as f:
            for line in f:
                word = ' '.join(line.strip().split()[1:])
                self.modal_verbs.add(word) # `join()` is used to port the modal verb with two or more tokens

    def build_pronouns(self):
        self.class2pronoun = {
            'PERSONAL_PRONOUN_SUBJECT': ['i', 'he', 'she', 'we', 'they'],
            'PERSONAL_PRONOUN_OBJECT': ['me', 'him', 'her', 'us', 'them'],
            'POSSESSIVE_DETERMINER': ['my', 'your', 'his', 'her', 'its', 'our', 'your', 'their'],
            'POSSESSIVE_PRONOUN': ['mine', 'yours', 'hers', 'ours', 'yours', 'theirs'],
            'REFLEXIVE_PRONOUN': ['myself', 'yourself', 'himself', 'itself', 'ourselves', 'yourselves', 'themselves']
        }

        self.pronoun2class = { pronoun: key for (key, value) in self.class2pronoun.items() for pronoun in value }
        self.pronouns = set(sum(list(self.class2pronoun.values()), [ ]))

    def update_corrupted_dataset(self, transformed_key, transformed_dict):
        for qid in self.data_dict:
            self.data_dict[qid]['corrupted_question'].update({
                transformed_key: transformed_dict[qid]
            })

    def save_corrupted_dataset(self):
        base_filename = os.path.split(self.dataset_filename)[1]
        main_filename, extension_filename = os.path.splitext(base_filename)
        target_main_filename = main_filename + '-corrupted'

        target_filename = os.path.join(self.target_dirname, target_main_filename + extension_filename)
        with open(target_filename, 'w') as f:
            json.dump(self.data_dict, f, indent = 4)

    def transform(self):
        paraphrased_dict = self.paraphrasing_transformation()
        self.update_corrupted_dataset('paraphrasing', paraphrased_dict)

        negated_dict = self.negation_transformation()
        self.update_corrupted_dataset('negation', negated_dict)

        conceptswapped_dict = self.conceptswap_transformation()
        self.update_corrupted_dataset('conceptswap', conceptswapped_dict)

        numberswapped_dict = self.numberswap_transformation()
        self.update_corrupted_dataset('numberswap', numberswapped_dict)

        datetimeswapped_dict = self.datetimeswap_transformation()
        self.update_corrupted_dataset('datetimeswap', datetimeswapped_dict)

        pronounswapped_dict = self.pronounswap_transformation()
        self.update_corrupted_dataset('pronounswap', pronounswapped_dict)

        self.save_corrupted_dataset()

class TestTransformer(SQUADTransformer):
    def __init__(self):
        super().__init__(None, None)

    def parse_input(self):
         self.data_dict = {
             'test0': {
                'context': 'xxxx',
                'gold_question': 'in what year did Common Sense begin publication?',
                'corrupted_question': { }
             }
         }

         self.sentences_dict = {
             'test0': 'in what year did Common Sense begin publication?'
         }

    def transform(self):
        conceptswaped_dict = self.conceptswap_transformation()
        numberswaped_dict = self.numberswap_transformation()
        datetimeswaped_dict = self.datetimeswap_transformation()
        pronounswaped_dict = self.pronounswap_transformation()
        print(conceptswaped_dict, numberswaped_dict, datetimeswaped_dict, pronounswaped_dict)

if __name__ == '__main__':
    t = TestTransformer()
    t.transform()