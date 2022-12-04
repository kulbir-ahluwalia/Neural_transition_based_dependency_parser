#!/usr/bin/env python
# coding: utf-8

# # CS 447 Homework 4 $-$ Dependency Parsing
# In this homework you will build a neural transition-based dependency parser, based off the paper <a href="https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf">A Fast and Accurate Dependency Parser using Neural Networks</a>.
# 
# The setup for a dependency parser is somewhat more sophisticated than tasks like classification or translation. Therfore, this homework contains many small functions that can be tested incrementally. In addition to Gradescope tests, we also provide substantial tests in the notebook that you can use to debug your code.
# 
# <font color='green'><b>Hint:</b> We suggest that you work on this homework in <b>CPU</b> until you are ready to train. At that point, you should switch your runtime to <b>GPU</b>. You can do this by going to <TT>Runtime > Change Runtime Type</TT> and select "GPU" from the dropdown menu.
# * You will find it easier to debug on CPU, and the error messages will be more understandable.
# * Google monitors your GPU usage and will occasionally restrict GPU access if you use it too much. In these cases, you can either switch to a different Google account or wait for your access to be restored.</font>

# # Step 0: Provided Testing Functions
# 
# The following cells set up the tests that you can use in the notebook. You should <b>not</b> edit any of these cells.

# In[1]:


### DO NOT EDIT ###

if __name__ == '__main__':
    import cloudpickle as cp
    from urllib.request import urlopen

    testing_bundle = cp.load(urlopen("https://drive.google.com/uc?export=download&id=1-9lWbXuXZYGjJWQRKD7um_83O-I3ER63"))


# In[2]:


### DO NOT EDIT ###

def sanityCheck(test_fxn, i=None, test_bundle=None, to_print = 'all', do_return=False, display_sent=True, do_raise=True):
    to_test = test_fxn.__name__ + (('_' + str(i)) if i is not None else '')
    if test_fxn.__name__ not in {'get_gold_action', 'get_lc', 'get_rc', 'get_top3_stack_features', 'get_top3_buffer_features'}: assert i is not None
    assert to_test in {'get_gold_action', 'get_lc', 'get_rc', 'get_top3_stack_features', 'get_top3_buffer_features', 'get_lc1_lc2_features_1',
                       'get_lc1_lc2_features_2', 'get_rc1_rc2_features_1', 'get_rc1_rc2_features_2', 'get_llc_rrc_features_1', 'get_llc_rrc_features_2'}
    assert to_print in {'all', 'incorrect', 'none'}
    test_bundle = testing_bundle if test_bundle is None else test_bundle
    sentences, configs, gold_actions, gold_features, _ = test_bundle
    vocab = Vocabulary(sentences)
    testsents = vocab.buildSentences(sentences)
    sents_correct, printag = 0, None # printag will collect the first error you encounter (used in Gradescope)
    for i in range(len(sentences)):

        printsent = "" # printsent is full printout for this sentence (only needed when to_print != 'none')
        sent, gold_deps = testsents[i][0], testsents[i][1]
        if to_test == 'get_gold_action':
            printsent += 'gold_dependencies: ' +str([str(arc) for arc in gold_deps.arcs]) + '\n\n'

        for j in range(len(configs[i])):
            # Manually create Stack, Buffer, Dependencies & ParserConfiguration objects from test case
            stk, buf, deps = configs[i][j]
            stack, buffer, dependencies = Stack([sent[idx] for idx in stk]), Buffer([sent[idx] for idx in buf]), Dependencies(vocab)
            for arc in deps: dependencies.add(sent[arc[0]], sent[arc[1]], arc[2])
            parser_config = ParserConfiguration(sent, vocab)
            parser_config.stack, parser_config.buffer, parser_config.dependencies = stack, buffer, dependencies

            if to_test == 'get_gold_action' or not (gold_actions[i][j] is None or gold_actions[i][j] == 'DONE'): # Don't need to test if Done or None when not testing get_gold_action

                # Call the student code
                arg = stack.get_si(1) if to_test in {'get_lc', 'get_rc'} else (int(to_test[-1]) if to_test[-1] in {'1', '2'} else None)
                if to_test == 'get_gold_action': fxn = lambda: test_fxn(stack, buffer, gold_deps)
                elif to_test in {'get_top3_stack_features', 'get_top3_buffer_features'}: fxn = lambda: test_fxn(parser_config)
                else: fxn = lambda: test_fxn(parser_config, arg)
                tt = to_test[:-2] if to_test[-1] in {'1', '2'} else to_test

                exception=None
                try:
                    yours = fxn()
                except Exception as e:
                    yours = "Raised Exception: " + str(e)
                    exception=e
                correct = gold_actions[i][j] if to_test == 'get_gold_action' else gold_features[i][j][to_test]
                fxn_name = 'get_gold_action(stack, buffer, gold_dependencies)' if to_test == 'get_gold_action'else  tt + ('(parser_config, 1)' if to_test not in to_test in {'get_lc', 'get_rc'} else '(parser_config, ' + str(arg) + ')')

                if to_test in {'get_lc', 'get_rc'} and exception is None:
                    if type(yours) != list or (len(yours) > 0 and type(yours[0]) != Arc):
                        yours = 'Your ' + to_test + '(...) did not return a list of Arcs' # note: exact quote used below, so if need to change, change in both places
                    else: yours = [str(arc) for arc in yours]
                # if random.random() < 0.05: yours = [None, None, None, None, None, None] # simulate getting it wrong
                is_correct = yours == correct

                # Make the printouts!
                printsent += 'Step ' + str(j+1) + ' | stack: ' + str([str(word) if to_test != 'get_gold_action' else word.word for word in stack]) + '\tbuffer: ' + str([str(word) if to_test != 'get_gold_action' else word.word for word in buffer]) + (('\tdependencies: ' + str([str(arc) for arc in dependencies.arcs])) if to_test != 'get_gold_action' else "") + '\n'
                printsent += '\tTesting '+ fxn_name +':\n'
                printsent += '\t\tYour Result: ' + str(yours) + '\t\tCorrect Result: ' + str(correct) + ('\t\tCorrect!' if is_correct else "") + '\n'
                if not is_correct:
                    printsent += '\t\tIncorrect! Halting parse of this sentence.\n\n'
                    if printag is None:
                        statement=yours if yours == 'Your ' + to_test + '(...) did not return a list of Arcs' else str(yours)[:20]+("... " if len(str(yours)) > 20 else "")
                        statement = "Your first error (on a hidden test sentence): You returned " + statement + "; expected " + str(correct) + "."
                        printag = statement if exception is None else (statement, exception)
                    if to_print != 'none':
                        print("Testing Sentence " + str(i+1) + '...')
                        if display_sent: display_sentence(sentences[i])
                        print(printsent)
                        print('Test Sentence ' + str(i+1) + ' failed.')
                        print(('\n' + '-'*100 + '\n'))
                    if do_raise and exception is not None:
                        raise exception
                    break
                else: printsent += '\n'

        else:
            sents_correct += 1
            if to_print == 'all':
                print("Testing Sentence " + str(i+1) + '...')
                if display_sent: display_sentence(sentences[i])
                print(printsent)
                print('Test Sentence ' + str(i+1) + ' passed!')
                print(('\n' + '-'*100 + '\n'))
    score = sents_correct / len(sentences)
    print(sents_correct, '/', len(sentences), '=', str(score*100)+'%', 'of test cases passed!')
    if do_return:
        return score, printag

def sanityCheck_generate_training_examples(test_fxn, feat_extract = None, test_bundle=None, to_print='all', do_return=False, display_sent=True, do_raise=True):
    assert test_fxn.__name__ == 'generate_training_examples' and (feat_extract is None or feat_extract.__name__ == 'extract_features')
    test_bundle = testing_bundle if test_bundle is None else test_bundle
    sentences, _, _, _, gold_examples = test_bundle
    vocab = Vocabulary(sentences)
    testsents = vocab.buildSentences(sentences)
    sents_correct, printag = 0, None
    for i in range(len(sentences)):
        sent, gold_deps = testsents[i][0], testsents[i][1]
        exception=None
        try:
            your_examples = test_fxn(sent, gold_deps, vocab, feat_extract=feat_extract if feat_extract is not None else lambda p: [])
        except Exception as e:
            your_examples = "Raised Exception: " + str(e)
            exception=e
        # if random.random() < 0.5: your_examples = [None, None, None, None, None, None, None, None] # simulate getting it wrong
        is_correct = your_examples == gold_examples[i][0 if feat_extract == None else 1]
        if to_print == 'all' or to_print == 'incorrect' and not is_correct:
            print("Testing Sentence " + str(i+1) + '...')
            if display_sent: display_sentence(sentences[i])
            print("sentence: " + str([word.word for word in sent]) + '\tgold_dependencies: ' + str([str(arc) for arc in gold_deps.arcs]))
            print("\tTesting generate_training_examples(sentence, gold_dependencies, vocab" + ('' if feat_extract is None else ', feat_extract=extract_features') + "):")
            print('\t\tYour Training Examples:   ', your_examples, '\n\t\tCorrect Training Examples:', gold_examples[i][0 if feat_extract == None else 1])
            print('\t\tCorrect!\n\n' if is_correct else '\t\tIncorrect!\n\n')
        if not is_correct and printag is None:
            statement = "Your first error (on a hidden test sentence): You returned " + str(your_examples)[0:20]+("... " if len(str(your_examples)) > 20 else "") + "; expected " + str(gold_examples[i]) + "."
            printag = statement if exception is None else (statement, exception)
        if to_print == 'all' or to_print == 'incorrect' and not is_correct:
            print(('Test Sentence ' + str(i+1) + ' passed!') if is_correct else ('Test Sentence ' + str(i+1) + ' failed.'))
            print('\n'+ '-'*100, '\n')
        if is_correct: sents_correct += 1
        if do_raise and exception is not None:
            raise exception

    score = sents_correct / len(sentences)
    print(sents_correct, '/', len(sentences), '=', str(score*100)+'%', 'of test cases passed!')
    if do_return:
        return score, printag

def sanityCheckStackBuffer():
    sanity_stack = Stack()
    item1 = Word('she', 'PRP', 0)
    item2 = Word('is', 'VBP', 1)
    item3 = Word('working', 'VBG', 2)
    sanity_stack.push(item1)
    sanity_stack.push(item2)
    sanity_stack.push(item3)
    p1 = False
    p2 = False
    p3 = False
    pop_item = sanity_stack.pop()
    if (pop_item.idx == 2 and pop_item.word == 'working' and pop_item.pos == 'VBG'):
        p1 = True
    pop_item = sanity_stack.pop()
    if (pop_item.idx == 1 and pop_item.word == 'is' and pop_item.pos == 'VBP'):
        p2 = True
    pop_item = sanity_stack.pop()
    if (pop_item.idx == 0 and pop_item.word == 'she' and pop_item.pos == 'PRP'):
        p3 = True
    if p1 and p2 and p3:
        print("Stack sanity check PASSED")
    else:
        print("Stack sanity check FAILED")

    sentence = [item1, item2, item3]
    sanity_buffer = Buffer(sentence)
    p1 = False
    p2 = False
    p3 = False
    pop_item = sanity_buffer.pop()
    if (pop_item.idx == 0 and pop_item.word == 'she' and pop_item.pos == 'PRP'):
        p1 = True
    pop_item = sanity_buffer.pop()
    if (pop_item.idx == 1 and pop_item.word == 'is' and pop_item.pos == 'VBP'):
        p2 = True
    pop_item = sanity_buffer.pop()
    if (pop_item.idx == 2 and pop_item.word == 'working' and pop_item.pos == 'VBG'):
        p3 = True
    if p1 and p2 and p3:
        print("Buffer sanity check PASSED")
    else:
        print("Buffer sanity check FAILED")

count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

def sanityCheckModel(all_test_params, model, expected_outputs, init_or_forward, batch_sizes=None):
    print('--- TEST: ' + ('Number of Model Parameters (tests __init__(...))' if init_or_forward=='init' else 'Output shape of forward(...)') + ' ---')

    for tp_idx, (test_params, expected_output) in enumerate(zip(all_test_params, expected_outputs)):       
        if init_or_forward == "forward":
            assert len(batch_sizes) == len(all_test_params)
            texts = torch.randint(1,4,(batch_sizes[tp_idx], test_params['n_features']))

        # Construct the student model
        tps = {k:v for k, v in test_params.items() if k != 'batch_size'}
        stu_parser_model = model(**tps)
        
        if init_or_forward == "forward":
            with torch.no_grad(): 
                stu_out = stu_parser_model(texts)
            ref_out_shape = expected_output

            has_passed = torch.is_tensor(stu_out)
            if not has_passed: msg = 'Output must be a torch.Tensor; received ' + str(type(stu_out))
            else: 
                has_passed = stu_out.shape == ref_out_shape
                msg = 'Your Output Shape: ' + str(stu_out.shape)
            

            status = 'PASSED' if has_passed else 'FAILED'
            message = '\t' + status + "\t Init Input: " + str({k:v for k,v in tps.items()}) + '\tForward Input Shape: ' + str(texts.shape) + '\tExpected Output Shape: ' + str(ref_out_shape) + '\t' + msg
            print(message)
        else:
            assert batch_sizes is None
            stu_num_params = count_parameters(stu_parser_model)
            ref_num_params = expected_output
            comparison_result = (stu_num_params == ref_num_params)

            status = 'PASSED' if comparison_result else 'FAILED'
            message = '\t' + status + "\tInput: " + str({k:v for k,v in test_params.items()}) + ('\tExpected Num. Params: ' + str(ref_num_params) + '\tYour Num. Params: '+ str(stu_num_params))
            print(message)

        del stu_parser_model


# # Step 1: Prepare Data
# 
# 
# 

# In[3]:


### DO NOT EDIT ###

import numpy as np
from spacy import displacy
import random


# ## Read & Visualize Data
# 
# We provide the data in a format preprocessed for this assignment. Here are some links if you are interested in learning more about the data:
# * We use one of the Universal Dependencies datasets from http://universaldependencies.org/docsv1/. Specifically, we use the UD_English dataset in version 1.4.
# * Refer to https://universaldependencies.org/ if you want to know more about the Universal Dependencies framework in general.
# * The data license can be found here: https://lindat.mff.cuni.cz/repository/xmlui/page/licence-UD-1.4
# 
# Run the following cells to load the data and see the number of sentences. You do <b>not</b> need to edit these cells.

# In[4]:


### DO NOT EDIT ###

def load_data():
    train_set = cp.load(urlopen("https://drive.google.com/uc?export=download&id=1N4B4bC4ua0bFMIYeNdtNQQ72LxxKLPas"))
    test_set = cp.load(urlopen("https://drive.google.com/uc?export=download&id=1TE2AflhABbz41dLMGmD1kTm7WevYpU31"))
    return train_set, test_set


# In[5]:


### DO NOT EDIT ###

if __name__ == '__main__':
    train_set, test_set = load_data()
    print("Num. Train Examples:", len(train_set))
    print("Num. Test Examples: ", len(test_set))


# Next, we visualize the training data, which contains labeled dependency trees. At test time, our goal will be to predict the dependency arcs & labels for a given sentence.

# In[6]:


### DO NOT EDIT ###

def display_sentence(sent):
    res = {'words': [{'text': "<ROOT>", 'tag': 'POS_ROOT'}], 'arcs': []}
    for i in range (len(sent['word'])):
        res['words'].append({'text': sent['word'][i], 'tag': sent['pos'][i]})
        s = i + 1
        e = sent['head'][i]
        direc = "left"
        if s > e: 
            s = sent['head'][i]
            e = i + 1
            direc = 'right'
        cur = {'start': s, 'end': e, 'label': sent['label'][i], 'dir': direc}
        res['arcs'].append(cur)
    displacy.render(res, style="dep", manual=True, jupyter=True, options={'distance': 70})


# In[7]:


if __name__ == '__main__':
    MIN_SENT_LEN, MAX_SENT_LEN = 4, 17 # You can change this if you're interested in seeing shorter/longer sentences
    for x in random.sample(list(filter(lambda x: len(x['word']) >= MIN_SENT_LEN and len(x['word']) <= MAX_SENT_LEN, train_set)), 5):
        display_sentence(x)


# ## Build Vocabulary
# 
# Next, we build the `Vocabulary` class. This maps each word, part-of-speech tag (POS), and label to an id (index), which we will later use in our embeddings. We also need to enumerate each possible transition, and map it to an id, since this is what our neural network will try to predict. The `Vocabulary` class does this as follows. Suppose there are $n$ labels. For each label, we create a Left-Arc (LA) and Right-Arc (RA) action for that label; and we also create a separate Shift (S) action. This creates a total of $2n+1$ actions that our dependency parser will be able to choose from.
# 
# You do <b>not</b> need to edit this cell.

# In[8]:


### DO NOT EDIT ###

class Vocabulary(object):
    def __init__(self, dataset):

        UNK = '<UNK>'
        NULL = '<NULL>'
        ROOT = '<ROOT>'

        # Find the label of the root
        root_labels = list([l for ex in dataset for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        assert len(set(root_labels)) == 1
        self.root_label = root_labels[0]

        # Create mapping from transitions to ids
        labels = sorted(list(set([w for ex in dataset for w in ex['label'] if w != self.root_label]))) # list of unique non-root labels
        labels = [self.root_label] + labels # add root label too
        self.n_labels = len(labels)

        transitions = ['LA-' + l for l in labels] + ['RA-' + l for l in labels] + ['S']
        self.n_trans = len(transitions) # 2*n_labels + 1
        self.tran2id = {t: i for (i, t) in enumerate(transitions)}
        self.id2tran = {i: t for (i, t) in enumerate(transitions)}

        # Create mapping from word, pos, & label to id
        # Do labels first
        self.LABEL_PREFIX = '<l>:'
        self.tok2id = {self.LABEL_PREFIX + l: i for (i, l) in enumerate(labels)}
        self.LABEL_NULL = self.tok2id[self.LABEL_PREFIX + NULL] = len(self.tok2id) # Add null label in

        # Do pos's
        self.POS_PREFIX = '<p>:'
        all_pos = sorted(set([self.POS_PREFIX + w for ex in dataset for w in ex['pos']])) # Get pos's
        self.tok2id.update({w: index + len(self.tok2id) for (index, w) in enumerate(all_pos)}) # Add poses in
        self.POS_NULL = self.tok2id[self.POS_PREFIX + NULL] = len(self.tok2id) # Add null pos
        self.POS_ROOT = self.tok2id[self.POS_PREFIX + ROOT] = len(self.tok2id) # Add root pos
        self.n_pos = 2 + len(all_pos) # +3 for null, root

        # Do words
        all_word = sorted(set([w for ex in dataset for w in ex['word']]))
        self.tok2id.update({w: index + len(self.tok2id) for (index, w) in enumerate(all_word)}) # Add words in
        self.WORD_UNK = self.tok2id[UNK] = len(self.tok2id) # Add unk word
        self.WORD_NULL = self.tok2id[NULL] = len(self.tok2id) # Add null word
        self.WORD_ROOT = self.tok2id[ROOT] = len(self.tok2id) # Add root word
        self.n_words = 3 + len(all_word) # +3 for unk, null, root

        self.id2tok = {v: k for (k, v) in self.tok2id.items()} # Flip it
        self.n_tokens = len(self.tok2id)

    def printStats(self):
        print('Num. labels:', self.n_labels)
        print('Num. transitions (2*n_labels + 1):', self.n_trans)
        print('Num. pos:', self.n_pos)
        print('Num. words:', self.n_words)
        print('Num. tokens:', self.n_tokens)


    def buildSentences(self, examples):
        processed_sentences = []
        for ex in examples:
            # Initialize words & dependencies
            words = [Word('<ROOT>', '<POS_ROOT>', 0, self.WORD_ROOT, self.POS_ROOT)]
            deps = []

            # Loop over words in sentence
            for i  in  range(len(ex['word'])):
                w = ex['word'][i]
                word_id = (self.tok2id[w] if w in self.tok2id else self.WORD_UNK)
                pos = self.POS_PREFIX + ex['pos'][i]
                pos_id = self.tok2id[pos]
                word = Word(ex['word'][i], ex['pos'][i],i+1, word_id, pos_id)

                words.append(word)
                deps.append((ex['head'][i], word, ex['label'][i] ))

            # Create dependencies
            dependencies = Dependencies(self)
            for dep in deps:
                dependencies.add(words[dep[0]], dep[1], dep[2])

            processed_sentences.append((words, dependencies))

        return processed_sentences


# Run the following cell to see some stats from the vocabulary.

# In[9]:


### DO NOT EDIT ###

if __name__ == '__main__':
    Vocabulary(train_set).printStats()


# # Step 2: Parser Data Structures [12 points]
# In this section, we define some useful data structures for dependency parsing. In particular, you will implement the `Stack` and `Buffer` structures, which make up a `ParserConfiguration`. You will also write a function to update these data structures based on a particular transition.

# ## Helpful Data Structures
# 
# First, we define some data classes that you will find useful. You will be working with these a lot, so you should understand the data they contain as well as their methods. You do <b>not</b> need to edit this cell.

# In[10]:


### DO NOT EDIT ###

class Word(object):
    '''
    Represents a word in the sentence.
    '''

    def __init__(self, word, pos, idx, word_id=None, pos_id=None):
        self.word = word
        self.pos = pos
        self.idx = idx
        self.word_id = word_id
        self.pos_id = pos_id

    def __str__(self):
        return 'Word(idx=' + str(self.idx) + ", word='" + self.word+"', pos='"+self.pos+"', word_id="+str(self.word_id)+', pos_id='+ str(self.pos_id) +')'

    def copy(self):
        return Word(self.word, self.pos, self.idx, self.word_id, self.pos_id)

    def __eq__(self, obj):
        if not isinstance(obj, Word): return False
        if obj.idx == self.idx:
            assert obj.word == self.word and obj.pos == self.pos and obj.word_id == self.word_id and obj.pos_id == self.pos_id, 'Your Word object has been corrupted.'
        return obj.idx == self.idx


class Arc(object):
    '''
    Represents an arc between two words.
    '''

    def __init__(self, head, dependent, label, label_id):
        self.head=head # Word object
        self.dependent=dependent # Word object
        self.label=label
        self.label_id = label_id

    def __str__(self):
        return 'Arc(head_idx='+str(self.head.idx)+', dep_idx='+str(self.dependent.idx)+', label_id='+ str(self.label_id)+')'


class Dependencies(object):
    '''
    Represents the dependency arcs in a sentence.
    '''

    def __init__(self, vocab):
        self.arcs = []
        self.vocab = vocab
        self.dep_to_head_mapping = {} # For fast lookup
    
    def add(self, head, dependent, label):
        '''
        Add a dependency from head to dependent with label.
        Inputs:
            head: Word object
            dependent: Word object
            label: str
        '''

        # comment it out later
        # print(f"head is: {head}")
        # print(f"dependent is: {dependent}")
        # print(f"label is: {label}")

        assert label[:3] != 'LA-' and label[:3] != 'RA-', 'You need to pass in just the label to add(...), not the entire action.'
        assert head is not None and dependent is not None, "You must pass in two Word objects to add(...)."

        self.arcs.append(Arc(head, dependent, label, self.vocab.tok2id[self.vocab.LABEL_PREFIX+label]))
        assert dependent.idx not in self.dep_to_head_mapping
        self.dep_to_head_mapping[dependent.idx] = self.arcs[-1]

    def getArcToHead(self, dependent):
        '''
        Returns the Arc that connects the head of dependent to dependent.
        Inputs:
            dependent: Word object
        '''
        if dependent.idx == 0: # Special case for ROOT
            return Arc(None, dependent, None, None)
        return self.dep_to_head_mapping[dependent.idx]

    def __iter__(self):
        return iter(self.arcs) # Allows you to iterate "for x in Dependencies(...)"


# ## <font color='red'>TODO</font>: Stack & Buffer [6 points]
# 
# Here, we provide you with the outline of stack and buffer data structures. Your task is to implement the `push(...)` and `pop(...)` methods of the `Stack`, and the `pop(...)` method of the `Buffer`. Each method is worth <b>2 points</b>.

# In[11]:


class Stack(object):
    def __init__(self, input=[]):
        '''
        Initialize an (empty) stack.
        '''
        self.stack = [word.copy() for word in input]
        # print(f"self.stack in init of Stack class is: {self.stack}")

    def push(self, item):
        '''
        Push item onto (the end of) self.stack. Returns nothing.
        '''


        self.stack.append(item)



        # print(f"Pushing item : {item}")
        #Pushing item : Word(idx=0, word='she', pos='PRP', word_id=None, pos_id=None)

        # print(f"Word object attributes : {item.__str__()}")
        # Word object attributes : Word(idx=0, word='she', pos='PRP', word_id=None, pos_id=None)

        # print(f"self.stack in Stack class push method is: {self.stack}")
        # self.stack in Stack class push method is: [<__main__.Word object at 0x7fb1b0b1d1f0>]

        return None


        # pass

    def pop(self):        
        '''
        Pop item from (the end of) self.stack. Returns the item popped.
        '''
        assert len(self.stack) > 0


        # print(f"self.stack before pop is: {self.stack}")
        popped_item = self.stack.pop()
        # print(f"popped_item is: {popped_item}")
        # print(f"self.stack after pop is: {self.stack}")

        # return None
        return popped_item

    def get_si(self, i):
        '''
        Returns si (the ith element of the stack) if it exists, otherwise None.
        '''
        assert i > 0, 'Must provide i > 0'
        return self.stack[-i] if len(self.stack) >= i else None

    def __getitem__(self, idx):
        return self.stack[idx]

    def __len__(self):
        return len(self.stack)

    def __str__(self):
        return str([str(x) for x in self.stack])


class Buffer(object):
    def __init__(self, sentence):
        '''
        Initialize as a list of words in sentence.
        '''
        self.buffer = [word.copy() for word in sentence]

    def pop(self):
        '''
        Pop item from (the beginning of) self.buffer. Returns the item popped.
        '''
        assert len(self.buffer) > 0

        # print(f"self.buffer before pop is: {self.buffer}")
        popped_buffer_item =  self.buffer.pop(0)


        # print(f"popped_item is: {popped_buffer_item}")
        # print(f"self.buffer after pop is: {self.buffer}")




        
        # return None
        return popped_buffer_item

    def get_bi(self, i):
        '''
        Returns bi (the ith element of the buffer) if it exists, otherwise None.
        '''
        assert i > 0, 'Must provide i > 0'
        return self.buffer[i-1] if len(self.buffer) >= i else None

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __len__(self):
        return len(self.buffer)

    def __str__(self):
        return str([str(x) for x in self.buffer])


# The code below runs a sanity check for your `Stack` and `Buffer` classes. The tests are similar to the hidden ones in Gradescope. However, note that passing the sanity check does <b>not</b> guarantee that you will pass the autograder; it is intended to help you debug.

# In[12]:


if __name__ == '__main__':
    sanityCheckStackBuffer()


# ## <font color='red'>TODO</font>: Parser Configuration [6 points]
# 
# Next, we create a `ParserConfiguration` class, which contains the `Stack`, `Buffer`, and `Dependencies` data structures. You only need to implement `parse_step(self, transition)`, which modifies these structures based on `transition`. This method is worth <b>6 points</b>.
# 
# More specifically, let $\sigma$ represent the stack, $\beta$ the buffer, and $A$ the set of arcs (dependencies). Based on the value of `transition`, `parse_step(self, transition)` should do the following:
# * `transition = 'S'`: &nbsp;<b>Shift</b> $w_k$ from the buffer to the stack. $(\sigma, w_k|\beta, A) \Rightarrow (\sigma|w_k, \beta, A)$
# * `transition = 'LA-label'`: &nbsp;Add a <b>left arc</b> with label $label$ from $w_j$ to $w_i$. $(\sigma |w_i w_j , \beta, A) \Rightarrow (\sigma |w_j, \beta, A \cup \{(w_j, label, w_i)\})$
# * `transition = 'RA-label'`: &nbsp;Add a <b>right arc</b> with label $label$ from $w_i$ to $w_j$. $(\sigma |w_i w_j , \beta, A) \Rightarrow (\sigma |w_i, \beta, A \cup \{(w_i, label, w_j)\})$
# 
# <font color='green'><b>Hint:</b> Use your `push(...)` and `pop(...)` operations here, and look at the methods of the `Dependencies` class to see how to add an arc to it.</font>

# In[13]:


class ParserConfiguration(object):
    def __init__(self, sentence, vocab):
        '''
        Inputs:
            sentence: list of Word objects
            vocab: Vocabulary object
        '''

        self.vocab = vocab

        assert sentence[0].word_id == self.vocab.WORD_ROOT
        self.stack = Stack([sentence[0]]) # Initialize stack with ROOT
        self.buffer = Buffer(sentence[1:]) # Initialize buffer with sentence

        # sentence in parser configuration is: [<__main__.Word object at 0x7f54916fb2b0>, <__main__.Word object at 0x7f54916fbf40>, <__main__.Word object at 0x7f54916fbfd0>, <__main__.Word object at 0x7f54916fb100>, <__main__.Word object at 0x7f54916fb9a0>, <__main__.Word object at 0x7f54916fb250>]
        # print(f"sentence in parser configuration is: {sentence}")

        # print(f"first word in sentence is: {sentence[0]}")
        # first word in sentence is: Word(idx=0, word='<ROOT>', pos='<POS_ROOT>', word_id=72, pos_id=33)
        # self.word = word
        # self.pos = pos
        # self.idx = idx
        # self.word_id = word_id
        # self.pos_id = pos_id


        self.dependencies = Dependencies(vocab)

        # # comment later later later on
        # dependencies_arcs_output = self.dependencies.arcs
        # dependencies_vocab_output = self.dependencies.vocab
        # dependencies_dep_to_head_mapping_output = self.dependencies.dep_to_head_mapping
        #
        # print(f"dependencies_arcs_output: {dependencies_arcs_output}")
        # print(f"dependencies_vocab_output: {dependencies_vocab_output}")
        # print(f"dependencies_dep_to_head_mapping_output: {dependencies_dep_to_head_mapping_output}")

        # self.dependencies: <__main__.Dependencies object at 0x7f549168bf40>
        # print(f"self.dependencies: {self.dependencies}")

        # gold_deps are: <__main__.Dependencies object at 0x7fb7e6e0ff10>
        # gold_deps_arcs_output: [<__main__.Arc object at 0x7fb7e6e0fc70>, <__main__.Arc object at 0x7fb7e9e9d760>, <__main__.Arc object at 0x7fb7e9e9dd30>, <__main__.Arc object at 0x7fb7e9e9de80>, <__main__.Arc object at 0x7fb7e9e9d910>, <__main__.Arc object at 0x7fb7e6dc3160>, <__main__.Arc object at 0x7fb7e6dc3be0>, <__main__.Arc object at 0x7fb7e6dc3220>, <__main__.Arc object at 0x7fb7e6dc3d00>]
        # gold_deps_vocab_output: <__main__.Vocabulary object at 0x7fb7e6e0f5e0>
        # gold_deps_dep_to_head_mapping_output: {1: <__main__.Arc object at 0x7fb7e6e0fc70>, 2: <__main__.Arc object at 0x7fb7e9e9d760>, 3: <__main__.Arc object at 0x7fb7e9e9dd30>, 4: <__main__.Arc object at 0x7fb7e9e9de80>, 5: <__main__.Arc object at 0x7fb7e9e9d910>, 6: <__main__.Arc object at 0x7fb7e6dc3160>, 7: <__main__.Arc object at 0x7fb7e6dc3be0>, 8: <__main__.Arc object at 0x7fb7e6dc3220>, 9: <__main__.Arc object at 0x7fb7e6dc3d00>}

    def parse_step(self, transition):
        #TODO: how can I test parse_Step?
        '''
        Update stack, buffer, and dependencies based on transition.
        Inputs:
            transition: str, "S", "LA-label", or "RA-label", where label is a valid label
        '''
        # print(f"self.vocab.tran2id : {self.vocab.tran2id}")
        # print(f"transition is: {transition}")

        assert transition in self.vocab.tran2id


        ### DONE ###
        # print(f"self.stack before pop is: {self.stack}")
        # popped_item = self.stack.pop()
        # print(f"popped_item is: {popped_item}")
        # print(f"self.stack after pop is: {self.stack}")

        # head is: Word(idx=3, word='had', pos='VBD', word_id=48, pos_id=26)
        # dependent is: Word(idx=2, word='news', pos='NN', word_id=55, pos_id=23)
        # label is: nsubj




        if transition[0] == "S":
            buffer_pop_output = self.buffer.pop()
            self.stack.push(buffer_pop_output)

        #DONE: How to check the following?? ==> Gradescope has the test for parse step
        elif transition[0] == "L":
            label = transition[3:]
            # w_i = self.stack.get_si(2)
            # w_j = self.stack.get_si(1)

            w_j = self.stack.pop()
            w_i = self.stack.pop()
            head = w_j
            dependent = w_i
            self.stack.push(w_j)

            #need to pop w_i
            # w_j_stack = self.stack.pop()
            # w_i_stack = self.stack.pop()
            # self.stack.push(w_j_stack)

            self.dependencies.add(head, dependent, label)



        elif transition[0] == "R":
            label = transition[3:]
            w_i = self.stack.get_si(2)
            w_j = self.stack.pop()
            head = w_i
            dependent = w_j

            # w_j_stack = self.stack.pop()

            self.dependencies.add(head, dependent, label)

        pass


# # Step 3: Generate Training Data [44 points]
# 
# As you saw above, the dataset contains many sentences along with their gold dependency parses. In order to use a transition-based parsing algorithm, we must predict the next parser action at each time step, based on the current parser configuration. Thus, we will need to transform each sentence into a series of training examples, where the input features are based on the current parser configuration and the correct label is the gold action.
# 
# In this section, you will first write an algorithm to select the next parser action at each time step based on the gold dependency parse. Then, you will extract features from each step's parser configuration, which the neural network will use to predict the next action.

# ## <font color='red'>TODO</font>: Compute Gold Action [8 points]
# 
# Next, you will write a function `get_gold_action(stack, buffer, gold_dependencies)`. Given a stack and buffer, this function should return the next action of the parser based on the gold dependencies. We encourage you to review the example of a sentence parsing in the lecture slides before attempting to implement this function. This method is worth <b>8 points</b>.
# 
# Let $s_i$ be $i$th element of the stack and $h(s_i)$ be the head word of $s_i$. The pseudocode is as follows:
# 
# 1. If the stack only contains `ROOT`:
#  - If the buffer is not empty, return `S` (shift).
#  - If the buffer is empty, return `DONE`, indicating the parse is complete.
# 2. If $h(s_2)=s_1$, return `LA-label`. Here, `label` is the label of the arc that attaches $s_2$ to $s_1$.
# 3. If $h(s_1)=s_2$ <b>and</b> $h(b_i) \neq s_1$ for all words $b_i$ in the buffer, return `RA-label`. Here, `label` is the label of the arc that attaches $s_1$ to $s_2$.
#  - This condition means that you cannot attach $s_1$ until everything in the buffer that depends on $s_1$ is attached. You should think about why this condition is necessary!
# 4. Otherwise:
#  - If the buffer is not empty, return `S`. 
#  - If the buffer is empty, return `None`, indicating a failed parse (i.e., the sentence is non-projective).
# 
# <font color='green'><b>Hint:</b> To get the $i$th word on the stack or buffer, call `stack.get_si(i)` or `buffer.get_bi(i)`. To find the head of a word `w`, call `gold_dependencies.getArcToHead(w)`.</font>

# In[14]:


def get_gold_action(stack, buffer, gold_dependencies):
    '''
    Given stack & buffer, compute the next gold action based on gold_dependencies.
    Args:
        - stack: Stack object
        - buffer: Buffer object
        - gold_dependencies: Dependencies object
    Returns:
        - action: str; 'S', 'LA-label', or 'RA-label', where 'label' is a valid label. Return None if no action possible and 'DONE' if the parse is complete.
    '''
    action = None

    # def __getitem__(self, idx):
    #     return self.stack[idx]
    #
    # def __len__(self):
    #     return len(self.stack)
    #
    # def __str__(self):
    #     return str([str(x) for x in self.stack])

    ### TODO ###
    # print(f"stack is :{str(stack)}")
    # print(f"buffer is :{str(buffer)}")
    s1 = stack.get_si(1)
    # print(f"s1 is: {s1}")

    s2 = stack.get_si(2)
    # print(f"s2 is: {s2}\n")





    if len(stack) == 1:     #TODO: len()
        if stack.__getitem__(0).word == '<ROOT>':  #TODO
            # if buffer.__len__() > 0:
            if len(buffer) > 0:
                # return 'S'
                # print(f"len of buffer {len(buffer)} > 0 and there is only root in the stack hence action = 'S'")
                action = 'S'
                return action

            # elif buffer.__len__() == 0:
            elif len(buffer) == 0:
                # return 'DONE'
                # print(f"len of buffer {len(buffer)} == 0 and there is only root in the stack hence action = 'DONE'")
                action = 'DONE'
                return action

    elif gold_dependencies.getArcToHead(s2).head == s1:
        arc_to_head = gold_dependencies.getArcToHead(s2)
        # print(f"arc_to_head in left action: {arc_to_head}")

        arc_to_head_label = arc_to_head.label
        # print(f"arc_to_head label in left action: {arc_to_head.label}")

        # sanity_buffer.get_bi(0): Word(idx=0, word='she', pos='PRP', word_id=None, pos_id=None)
        # sanity_buffer.__getitem__(1): Word(idx=1, word='is', pos='VBP', word_id=None, pos_id=None)

        # return 'LA-' + arc_to_head_label
        action = 'LA-' + arc_to_head_label
        return action

    # If $h(s_1)=s_2$ <b>and</b> $h(b_i) \neq s_1$ for all words $b_i$ in the buffer, return `RA-label`. Here, `label` is the label of the arc that attaches $s_1$ to $s_2$.
    #  - This condition means that you cannot attach $s_1$ until everything in the buffer that depends on $s_1$ is attached. You should think about why this condition is necessary!

    elif gold_dependencies.getArcToHead(s1).head == s2:
        h_of_b_i_list = []
        # print(f"h_of_b_i  in right arc is : {h_of_b_i}")

        for index, b_i in enumerate(buffer):
            # print(f"b_i is: {b_i}")
            # print(f"head of b_i is: {gold_dependencies.getArcToHead(b_i).head}")
            # print(f"is head of b_i == s1? : {gold_dependencies.getArcToHead(b_i).head == s1}")
            if gold_dependencies.getArcToHead(b_i).head == s1:
                h_of_b_i = False
                h_of_b_i_list.append(1)
                # print(f"h_of_b_i ==s1 and hence flag is : {h_of_b_i}")
            elif gold_dependencies.getArcToHead(b_i).head != s1:
                h_of_b_i_list.append(0)

        h_of_b_i_list_sum = sum(h_of_b_i_list)
        # print(f"h_of_b_i_list: {h_of_b_i_list}")
        # print(f"h_of_b_i_list sum is: {h_of_b_i_list_sum}")

        if gold_dependencies.getArcToHead(s1).head == s2 and (h_of_b_i_list_sum==0):
            arc_to_head = gold_dependencies.getArcToHead(s1)
            # print(f"arc_to_head in right arc: {arc_to_head}")

            arc_to_head_label = arc_to_head.label
            # print(f"arc_to_head label  in right arc: {arc_to_head.label}")

            # sanity_buffer.get_bi(0): Word(idx=0, word='she', pos='PRP', word_id=None, pos_id=None)
            # sanity_buffer.__getitem__(1): Word(idx=1, word='is', pos='VBP', word_id=None, pos_id=None)

            # return 'RA-' + arc_to_head_label
            action = 'RA-' + arc_to_head_label
            return action

    # 4. Otherwise:
    #  - If the buffer is not empty, return `S`.
    #  - If the buffer is empty, return `None`, indicating a failed parse (i.e., the sentence is non-projective).

    if len(buffer) != 0:
        # return 'S'
        # print(f"len of buffer is greater than zero :{len(buffer)} and hence action = 'S'")
        action = 'S'
        return action

    if len(buffer) == 0:
        # return 'None'
        # print(f"len of buffer is zero :{len(buffer)} and hence action = None")
        action = None
        # action = 'S'
        return action

    # else:
    #     print(f"DID NOT go in any if condition")

    
    return action


# We provide you with 10 sentences for a sanity check of this function. The first sentence is the example from the lecture slides, the next 8 sentences are artificial sentences designed to test edge cases, and the last sentence is an example from the training set. For each sentence, we have hard-coded the stack & buffer configurations that you should encounter as well as the correct action.
# 
# The `to_print` argument of this method may be set to one of the following values:
# * `all`: Print every sentence.
# * `incorrect`: Print only the sentences that your function gets incorrect.
# * `none`: Don't print any of the sentences (only show the score).
# 
# You may also toggle `do_raise`, which controls whether an exception in your code is raised (halting execution) or suppressed (thus allowing all test sentences to complete so you can see your score).
# 
# Note that Gradescope uses a set of different (hidden) tests, so you will want to fully test your code here.

# In[14]:





# In[15]:


if __name__ == '__main__':
    # sanityCheck(get_gold_action, to_print='incorrect', do_raise=True)
    sanityCheck(get_gold_action, to_print='all', do_raise=True)


# ## <font color='red'>TODO</font>: Generate Training Examples [8 points]
# 
# 
# Now you will write a function to generate the training examples. Recall that each sentence needs to be converted into a series of separate training examples. Each training example will essentially be a partial parser configuration along with its gold action; the goal of the neural network will be to predict this action from the parser configuration.
# 
# In order to make this prediction, you need to extract features from the parser configuration. You will implement the feature extraction method in a future section; for now, we pass in a dummy function `feat_extract(parser_config)` that returns an empty feature list.
# 
# This function is worth <b>8 points</b>.

# In[16]:


def generate_training_examples(sentence, gold_dependencies, vocab, feat_extract = lambda parser_config: []):
    '''
    Create training instances for sentence.
    Inputs:
        sentence: list of Word objects
        gold_dependencies: Dependencies object that contains the complete gold dependency tree
        vocab: Vocabulary object
        feat_extract: Feature extraction function
    Outputs:
        training_examples: List of tuples (features, label), where features is a list and label is a string
    Pseudocode:
        (1) Initialize your parser configuration (note that the __init__ method of ParserConfiguration creates the stack & buffer for you)
        (2) Repeatedly call get_gold_action(...) on your current parser confuration until the gold action is 'DONE'
        (3) If the gold action is None at any step, return []  (indicating the sentence cannot be parsed; it is non-projective)
        (4) Otherwise, append tuple (features, gold action) to training_examples, where features is the result of calling feat_extract on your parser configuration
        (5) Update your parser configuration according to the gold action
        (6) Return training_examples
    '''

    training_examples = []

    parser_config = ParserConfiguration(sentence, vocab)
    stack = parser_config.stack
    buffer = parser_config.buffer
    # parser_config.dependencies = gold_dependencies

    # def get_gold_action(stack, buffer, gold_dependencies):
    # '''
    # Given stack & buffer, compute the next gold action based on gold_dependencies.
    # Args:
    #     - stack: Stack object
    #     - buffer: Buffer object
    #     - gold_dependencies: Dependencies object
    # Returns:
    #     - action: str; 'S', 'LA-label', or 'RA-label', where 'label' is a valid label. Return None if no action possible and 'DONE' if the parse is complete.
    # '''
    while get_gold_action(stack, buffer, gold_dependencies) != 'DONE':
        gold_action =  get_gold_action(stack, buffer, gold_dependencies)
        # print(f"gold_action outside none check: {gold_action}")


        #Use is when you want to check against an object's identity (e.g. checking to see if var is None). Use == when you want to check equality
        if get_gold_action(stack, buffer, gold_dependencies) is None:
            # gold_action =  get_gold_action(stack, buffer, gold_dependencies)
            # print(f"gold_action in NONE check: {gold_action}")
            return []

        elif get_gold_action(stack, buffer, gold_dependencies) is not None:
            # gold_action =  get_gold_action(stack, buffer, gold_dependencies)
            # print(f"gold_action in != NONE check: {gold_action}")
            features = feat_extract(parser_config)
            training_examples.append((features, gold_action))
            parser_config.parse_step(gold_action)

    return training_examples


# We provide you with a sanity check for this function on the same test sentences we used above.

# In[17]:


if __name__ == '__main__':
    sanityCheck_generate_training_examples(generate_training_examples, to_print='incorrect', do_raise=True)


# The following function calls `generate_training_examples(...)` on every sentence in the dataset to create the full training data. You do <b>not</b> need to edit it.

# In[18]:


### DO NOT EDIT ###

def generate_all_training_examples(vocab, sentences, feat_extract = lambda parser_config: []):
    '''
    Generate training examples for all sentences.
    '''
    all_training_examples = []
    successful_sents = 0
    for sentence in sentences:
        training_examples = generate_training_examples(sentence[0], sentence[1], vocab, feat_extract)
        if training_examples != []:
            all_training_examples += training_examples
            successful_sents += 1

    print("Successfully generated training examples for", successful_sents, "/", len(sentences), "sentences")
    print("Number of training examples:", len(all_training_examples))

    return all_training_examples


# In[19]:


### DO NOT EDIT ###

if __name__ == '__main__':
    _vocab = Vocabulary(train_set) # Variable just used in this cell
    generate_all_training_examples(_vocab, _vocab.buildSentences(train_set))


# ## <font color='red'>TODO</font>: Extract Features [28 points]
# By this point, you have written code to create individual training instances. Each instance is made up of a parser configuration along with the gold action that the classifier should be trained to predict.
# 
# In order to make this prediction, your neural network will have to rely on features extracted from each parser configuration. We follow the procedure described at the end of Section 3.1 of <a href="https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf">A Fast and Accurate Dependency Parser using Neural Networks</a>. In total, we will extract 48 features from the parser configuration $-$ 18 word features, 18 POS features, and 12 label features:
# * Word & POS features for $s_1$, $s_2$, $s_3$ (top 3 items of the stack)
# * Word & POS features for $b_1$, $b_2$, $b_3$ (top 3 items of the buffer)
# * Word, POS, & label features for $lc_1(s_1)$, $lc_2(s_1)$, $lc_1(s_2)$, $lc_2(s_2)$ (the first & second leftmost children of the top 2 items on the stack)
# * Word, POS, & label features for $rc_1(s_1)$, $rc_2(s_1)$, $rc_1(s_2)$, $rc_2(s_2)$ (the first & second rightmost children of the top 2 items on the stack)
# * Word, POS, & label features for $lc_1(lc_1(s_1))$, $lc_1(lc_1(s_2))$, $rc_1(rc_1(s_1))$, $rc_1(rc_1(s_2))$ (the leftmost of the leftmost & rightmost of the rightmost children of the top 2 items on the stack)
# 
# You will write a separate function for each of the 5 bullets above. Each function will return a list of word features, a list of POS features, and (in the relevant cases) a list of label features. We also provide you with a test for each function.
# 
# A "feature" refers to the id (index in the Vocabulary) of a word, POS, or label. Your neural network will then construct embeddings for each id, much as you have seen in previous homeworks.

# First, write 2 functions to extract features corresponding to the words at the top of the stack & buffer:
# * `get_top3_stack_features(parser_config)` <b>[3 points]</b>: Return word & POS features (ids) for $s_1$, $s_2$, $s_3$ (top 3 words on the stack).
# * `get_top3_buffer_features(parser_config)` <b>[3 points]</b>: Return word & POS features (ids) for $b_1$, $b_2$, $b_3$ (top 3 words on the buffer).
# 
# Wherever a particular word does not exist (such as when the stack or buffer has length $< 3$) use the appropriate NULL token. This is necessary to ensure that our neural network will see an equally sized feature vector for each example.

# In[20]:


def get_top3_stack_features(parser_config):
    '''
    Get the word and POS features for s1, s2, and s3 (the top 3 items on the stack)
    Returns:
        word_features: List of word ids for s1, s2, s3 (use vocab.WORD_NULL if a word does not exist)
        pos_features: List of POS ids for s1, s2, s3 (use vocab.POS_NULL if a word does not exist)
    '''
    word_features, pos_features = [parser_config.vocab.WORD_NULL]*3, [parser_config.vocab.POS_NULL]*3
    # print(f"word_features before processing: {word_features}")
    # print(f"pos_features before processing: {pos_features}")

    ### TODO ###
    # test = parser_config.stack.get_si(1)
    #
    # print(f"test is {test.word_id}")

    for i in range(1,4):
        if parser_config.stack.get_si(i) is not None:
            s_i = (parser_config.stack.get_si(i))
            word_features[i-1] = s_i.word_id
            pos_features[i-1] = s_i.pos_id
        elif parser_config.stack.get_si(i) is None:
            pass

    # for a word:
    # self.word = word
    # self.pos = pos
    # self.idx = idx
    # self.word_id = word_id
    # self.pos_id = pos_id

    # print(f"word_features after processing: {word_features}")
    # print(f"pos_features after processing: {pos_features}")
    #
    # return None, None
    return word_features, pos_features


# In[21]:


if __name__ == '__main__':
    sanityCheck(get_top3_stack_features, to_print='incorrect', do_raise=True)


# In[22]:


def get_top3_buffer_features(parser_config):
    '''
    Get the word and POS features for b1, b2, and b3 (the top 3 items on the buffer)
    Returns:
        word_features: List of word ids for b1, b2, b3 (use vocab.WORD_NULL if a word does not exist)
        pos_features: List of POS ids for b1, b2, b3 (use vocab.POS_NULL if a word does not exist)
    '''
    word_features, pos_features = [parser_config.vocab.WORD_NULL]*3, [parser_config.vocab.POS_NULL]*3

    ### TODO ###
    for i in range(1,4):
        if parser_config.buffer.get_bi(i) is not None:
            b_i = (parser_config.buffer.get_bi(i))
            word_features[i-1] = b_i.word_id
            pos_features[i-1] = b_i.pos_id
        elif parser_config.buffer.get_bi(i) is None:
            pass

    
    return word_features, pos_features


# In[23]:


if __name__ == '__main__':
    sanityCheck(get_top3_buffer_features, to_print='incorrect', do_raise=True)


# The remaining features have to do with the leftmost & rightmost children of the words at the top of the stack & buffer. Write the following 2 helper functions to make it easier to access these dependents:
# * `get_lc(parser_config, word)` <b>[2 points]</b>: Return a list of arcs to dependents of `word`, sorted from <b>left to right</b>. Only include dependents that are to the <b>left</b> of `word` in the sentence.
# * `get_rc(parser_config, word)` <b>[2 points]</b>: Return a list of arcs to dependents of `word`, sorted from <b>right to left</b>. Only include dependents that are to the <b>right</b> of `word` in the sentence.
# 
# <font color='green'><b>Hint:</b> You can sort a list of objects using `sorted(...)` with the `key` parameter.</font>
# 
# <font color='green'><b>Hint:</b> Each of these functions can be written in as few as 1 line. If you find yourself using more than 5 lines, you are probably doing more work than you need to.</font>

# In[24]:


def get_lc(parser_config, word):
    '''
    Finds the left dependents of word, sorted from left to right.
    Returns:
        A list of Arcs whose head is word, sorted by the indices of the dependent word from left to right.
    '''
    ### TODO ###

    # gold_dependencies.getArcToHead(s2).head

    # look at the dependencies
    # print(f"parser_config.dependencies: {parser_config.dependencies}")

    arc_list = []

    for dependency in parser_config.dependencies:
        if dependency.head.idx == word.idx:
            if dependency.head.idx > dependency.dependent.idx:
                arc_list.append(dependency)

    # lambda arguments : expression
    key_function = lambda arc: arc.dependent.idx

    # alternate way
    # def key_function(arc):
    #     return arc.dependent.idx

    arc_list_sorted = sorted(arc_list, key = key_function)


    # look at the first half of the sentence before the current word_idx

    return arc_list_sorted


# In[25]:


if __name__ == '__main__':
    sanityCheck(get_lc, to_print='incorrect', do_raise=True)


# In[26]:


def get_rc(parser_config, word):
    '''
    Finds the right dependents of word, sorted from right to left.
    Returns:
        A list of Arcs whose head is word, sorted by the indices of the dependent word from right to left.
    '''
    ### TODO ###

    arc_list = []

    for dependency in parser_config.dependencies:
        if dependency.head.idx == word.idx:
            if dependency.head.idx < dependency.dependent.idx:
                arc_list.append(dependency)

    # lambda arguments : expression
    key_function = lambda arc: arc.dependent.idx

    # alternate way
    # def key_function(arc):
    #     return arc.dependent.idx

    arc_list_sorted = sorted(arc_list, key = key_function)

    #Using reversed() we can reverse the list and a list_reverseiterator object is created, from which we can create a list using list() type casting. Or, we can also use list.reverse() function to reverse list in-place.
    #or do slicing [::-1]. but a new copy is created. This exhausts more memory.
    list_of_arcs_reversed = (arc_list_sorted[::-1])
    # print(f"list_of_arcs_reversed: {list_of_arcs_reversed}")
    # print(f"type of list_of_arcs_reversed: {type(list_of_arcs_reversed)}")

    # return None
    return list_of_arcs_reversed


# In[27]:


if __name__ == '__main__':
    sanityCheck(get_rc, to_print='incorrect', do_raise=True)


# Let $lc_j(s_i)$ be the $j$th leftmost child of the $i$th item on the stack. Write the following function:
# * `get_lc1_lc2_features(parser_config, i)` <b>[6 points]</b>: Return word & POS features for $lc_1(s_i)$ and $lc_2(s_i)$. Additionally, return label features (the label ids) for the arcs that attach $lc_1(s_i)$ to $s_i$ and $lc_2(s_i)$ to $s_i$. As before, wherever a particular word does not exist, use the appropriate NULL token.
# 
# We will call this function with `i=1` and `i=2`, accounting for the words $lc_1(s_1)$, $lc_2(s_1)$, $lc_1(s_2)$, $lc_2(s_2)$.

# In[28]:


def get_lc1_lc2_features(parser_config, i):

    '''
    Get the word, POS, and label features for lc1(si) and lc2(si), where i in {1, 2}
    Returns:
        word_features: List of word ids for lc1(si), lc2(si) (use vocab.WORD_NULL if a word does not exist)
        pos_features: List of POS ids for lc1(si), lc2(si) (use vocab.POS_NULL if a word does not exist)
        label_features: List of label ids for lc1(si), lc2(si) (use vocab.LABEL_NULL if a word does not exist)
    '''
    assert i in {1,2}
    word_features, pos_features, label_features = [parser_config.vocab.WORD_NULL]*2, [parser_config.vocab.POS_NULL]*2, [parser_config.vocab.LABEL_NULL]*2

    ### TODO ###
    # def get_lc(parser_config, word):
    # '''
    # Finds the left dependents of word, sorted from left to right.
    # Returns:
    #     A list of Arcs whose head is word, sorted by the indices of the dependent word from left to right.
    # '''

    # def get_rc(parser_config, word):
    # '''
    # Finds the right dependents of word, sorted from right to left.
    # Returns:
    #     A list of Arcs whose head is word, sorted by the indices of the dependent word from right to left.
    # '''


    # class Arc(object):
    # '''
    # Represents an arc between two words.
    # '''
    #
    #     def __init__(self, head, dependent, label, label_id):
    #         self.head=head # Word object
    #         self.dependent=dependent # Word object
    #         self.label=label
    #         self.label_id = label_id
    #
    #     def __str__(self):
    #         return 'Arc(head_idx='+str(self.head.idx)+', dep_idx='+str(self.dependent.idx)+', label_id='+ str(self.label_id)+')'


    # for i in range(1,3):
    if parser_config.stack.get_si(i) is not None:

        s_i = (parser_config.stack.get_si(i))

        arc_list_sorted = get_lc(parser_config, s_i)
        # print(f"arc_list_sorted: {arc_list_sorted}")


        jth_leftmost_child_arcs = arc_list_sorted[:2]
        # print(f"jth_leftmost_child_arcs: {jth_leftmost_child_arcs}")

        jth_leftmost_child_words = [arc.dependent for arc in jth_leftmost_child_arcs]
        jth_leftmost_child_labels = [arc.label_id for arc in jth_leftmost_child_arcs]

        # print(f"jth_leftmost_child_words: {jth_leftmost_child_words}")
        # print(f"jth_leftmost_child_labels: {jth_leftmost_child_labels}")


        if len(jth_leftmost_child_words) >= 1:
            # for k in range (1,2):
            k = 1
            word_object = jth_leftmost_child_words[k-1]
            word_features[k-1] = word_object.word_id
            pos_features[k-1] = jth_leftmost_child_words[k-1].pos_id
            label_features[k-1] = jth_leftmost_child_labels[k-1]

            if len(jth_leftmost_child_words) >= 2:
                k = 2
                word_object = jth_leftmost_child_words[k-1]
                word_features[k-1] = word_object.word_id
                pos_features[k-1] = jth_leftmost_child_words[k-1].pos_id
                label_features[k-1] = jth_leftmost_child_labels[k-1]

    elif parser_config.stack.get_si(i) is None:
        pass


    
    return word_features, pos_features, label_features


# In[29]:


if __name__ == '__main__':
    sanityCheck(get_lc1_lc2_features, i=1, to_print='incorrect', do_raise=True) # call with i=1


# In[30]:


if __name__ == '__main__':
    sanityCheck(get_lc1_lc2_features, i=2, to_print='incorrect', do_raise=True) # call with i=2


# You will now write the analagous function for the rightmost children. Let $rc_j(s_i)$ be the $j$th rightmost child of the $i$th item on the stack. Write the following function:
# * `get_rc1_rc2_features(parser_config, i)` <b>[6 points]</b>: Return word & POS features for $rc_1(s_i)$ and $rc_2(s_i)$. Additionally, return label features (the label ids) for the arcs that attach $rc_1(s_i)$ to $s_i$ and $rc_2(s_i)$ to $s_i$. As before, wherever a particular word does not exist, use the appropriate NULL token.
# 
# We will call this function with `i=1` and `i=2`, accounting for the words $rc_1(s_1)$, $rc_2(s_1)$, $rc_1(s_2)$, $rc_2(s_2)$.

# In[31]:


def get_rc1_rc2_features(parser_config, i):
    '''
    Get the word, POS, and label features for rc1(si) and rc2(si), where i in {1, 2}
    Returns:
        word_features: List of word ids for rc1(si), rc2(si) (use vocab.WORD_NULL if a word does not exist)
        pos_features: List of POS ids for rc1(si), rc2(si) (use vocab.POS_NULL if a word does not exist)
        label_features: List of label ids for rc1(si), rc2(si) (use vocab.LABEL_NULL if a word does not exist)
    '''
    assert i in {1,2}
    word_features, pos_features, label_features = [parser_config.vocab.WORD_NULL]*2, [parser_config.vocab.POS_NULL]*2, [parser_config.vocab.LABEL_NULL]*2

    ### TODO ###
    if parser_config.stack.get_si(i) is not None:

        s_i = (parser_config.stack.get_si(i))

        arc_list_sorted = get_rc(parser_config, s_i)
        # print(f"arc_list_sorted: {arc_list_sorted}")


        jth_leftmost_child_arcs = arc_list_sorted[:2]
        # print(f"jth_leftmost_child_arcs: {jth_leftmost_child_arcs}")

        jth_rightmost_child_words = [arc.dependent for arc in jth_leftmost_child_arcs]
        jth_rightmost_child_labels = [arc.label_id for arc in jth_leftmost_child_arcs]

        # print(f"jth_rightmost_child_words: {jth_rightmost_child_words}")
        # print(f"jth_leftmost_child_labels: {jth_rightmost_child_labels}")


        if len(jth_rightmost_child_words) >= 1:
            # for k in range (1,2):
            k = 1
            word_object = jth_rightmost_child_words[k-1]
            word_features[k-1] = word_object.word_id
            pos_features[k-1] = jth_rightmost_child_words[k-1].pos_id
            label_features[k-1] = jth_rightmost_child_labels[k-1]

            if len(jth_rightmost_child_words) >= 2:
                k = 2
                word_object = jth_rightmost_child_words[k-1]
                word_features[k-1] = word_object.word_id
                pos_features[k-1] = jth_rightmost_child_words[k-1].pos_id
                label_features[k-1] = jth_rightmost_child_labels[k-1]

    elif parser_config.stack.get_si(i) is None:
        pass



    return word_features, pos_features, label_features


# In[32]:


if __name__ == '__main__':
    sanityCheck(get_rc1_rc2_features, i=1, to_print='incorrect', do_raise=True) # call with i=1


# In[33]:


if __name__ == '__main__':
    sanityCheck(get_rc1_rc2_features, i=2, to_print='incorrect', do_raise=True) # call with i=2


# Finally, write the following function:
# * `get_llc_rrc_features(parser_config, i)` <b>[6 points]</b>: Return word & POS features for $lc_1(lc_1(s_i))$ and $rc_1(rc_1(s_i))$. Additionally, return label features (the label ids) for the arcs that attach $lc_1(lc_1(s_i))$ to $lc_1(s_i)$ and $rc_1(rc_1(s_i))$ to $rc_1(s_i)$. As before, wherever a particular word does not exist, use the appropriate NULL token.
# 
# We will call this function with `i=1` and `i=2`, accounting for the words $lc_1(lc_1(s_1))$, $lc_1(lc_1(s_2))$, $rc_1(rc_1(s_1))$, $rc_1(rc_1(s_2))$.

# In[34]:


def get_llc_rrc_features(parser_config, i):
    '''
    Get the word, POS, and label features for lc1(lc1(si)), and rc1(rc1(si)), where i in {1, 2}
    Returns:
        word_features: List of word ids for lc1(lc1(si)), and rc1(rc1(si)) (use vocab.WORD_NULL if a word does not exist)
        pos_features: List of POS ids for lc1(lc1(si)), and rc1(rc1(si)) (use vocab.POS_NULL if a word does not exist)
        label_features: List of label ids for lc1(lc1(si)), and rc1(rc1(si)) (use vocab.LABEL_NULL if a word does not exist)
    '''
    assert i in {1,2}
    word_features, pos_features, label_features = [parser_config.vocab.WORD_NULL]*2, [parser_config.vocab.POS_NULL]*2, [parser_config.vocab.LABEL_NULL]*2

    ### TODO ###
    if parser_config.stack.get_si(i) is not None:

        s_i = (parser_config.stack.get_si(i))

        arc_list_sorted_rc = get_rc(parser_config, s_i)
        arc_list_sorted_lc = get_lc(parser_config, s_i)

        # print(f"arc_list_sorted: {arc_list_sorted}")
        jth_rightmost_child_arcs = arc_list_sorted_rc[:1]
        jth_leftmost_child_arcs = arc_list_sorted_lc[:1]

        # print(f"jth_leftmost_child_arcs: {jth_leftmost_child_arcs}")
        jth_rightmost_child_words = [arc.dependent for arc in jth_rightmost_child_arcs]
        jth_rightmost_child_labels = [arc.label_id for arc in jth_rightmost_child_arcs]
        # print(f"jth_rightmost_child_words: {jth_rightmost_child_words}")
        # print(f"jth_leftmost_child_labels: {jth_rightmost_child_labels}")

        jth_leftmost_child_words = [arc.dependent for arc in jth_leftmost_child_arcs]
        jth_leftmost_child_labels = [arc.label_id for arc in jth_leftmost_child_arcs]
        # print(f"jth_leftmost_child_words: {jth_leftmost_child_words}")
        # print(f"jth_leftmost_child_labels: {jth_leftmost_child_labels}")

        if len(jth_rightmost_child_words) >=1:
            arc_list_sorted_2 = get_rc(parser_config, jth_rightmost_child_words[0])
            jth_rightmost_child_arcs_2 = arc_list_sorted_2[:1]
            jth_rightmost_child_words_2 = [arc.dependent for arc in jth_rightmost_child_arcs_2]
            jth_rightmost_child_labels_2 = [arc.label_id for arc in jth_rightmost_child_arcs_2]
            #
            # print(f"jth_rightmost_child_words_2: {jth_rightmost_child_words_2}")
            # print(f"jth_rightmost_child_labels_2: {jth_rightmost_child_labels_2}")


            if len(jth_rightmost_child_words_2) >= 1:
                # for k in range (1,2):
                # k = 1
                word_object = jth_rightmost_child_words_2[0]
                word_features[1] = word_object.word_id
                pos_features[1] = jth_rightmost_child_words_2[0].pos_id
                label_features[1] = jth_rightmost_child_labels_2[0]



        if len(jth_leftmost_child_words) >=1:
            arc_list_sorted_2_lc = get_lc(parser_config, jth_leftmost_child_words[0])
            jth_leftmost_child_arcs_2 = arc_list_sorted_2_lc[:1]
            jth_leftmost_child_words_2 = [arc.dependent for arc in jth_leftmost_child_arcs_2]
            jth_leftmost_child_labels_2 = [arc.label_id for arc in jth_leftmost_child_arcs_2]

            # print(f"jth_leftmost_child_words_2: {jth_leftmost_child_words_2}")
            # print(f"jth_leftmost_child_labels_2: {jth_leftmost_child_labels_2}")


            if len(jth_leftmost_child_words_2) >= 1:
                # for k in range (1,2):
                # k = 1
                word_object = jth_leftmost_child_words_2[0]
                word_features[0] = word_object.word_id
                pos_features[0] = jth_leftmost_child_words_2[0].pos_id
                label_features[0] = jth_leftmost_child_labels_2[0]


                # if len(jth_rightmost_child_words) >= 2:
                #     k = 2
                #     word_object = jth_rightmost_child_words[k-1]
                #     word_features[k-1] = word_object.word_id
                #     pos_features[k-1] = jth_rightmost_child_words[k-1].pos_id
                #     label_features[k-1] = jth_rightmost_child_labels[k-1]

    elif parser_config.stack.get_si(i) is None:
        pass



    return word_features, pos_features, label_features
    


# In[35]:


if __name__ == '__main__':
    sanityCheck(get_llc_rrc_features, i=1, to_print='incorrect', do_raise=True) # call with i=1


# In[36]:


if __name__ == '__main__':
    sanityCheck(get_llc_rrc_features, i=2, to_print='incorrect', do_raise=True) # call with i=2


# We provide you with a function `extract_features(parser_config)` that calls each of these functions and returns a list of the 48 total features. You do <b>not</b> need to edit this function.

# In[37]:


### DO NOT EDIT ###

def extract_features(parser_config): # for both train & inference
    word_features, pos_features, label_features = [], [], []

    # 1. Get word & pos features for s1, s2, and s3
    (x, y) = get_top3_stack_features(parser_config)
    word_features, pos_features = word_features + x, pos_features + y


    # 2. Get word & pos features for b1, b2, and b3
    (x, y) = get_top3_buffer_features(parser_config)
    word_features, pos_features = word_features + x, pos_features + y


    # 3. Get word & pos & label features for lc1(s1), lc1(s2), lc2(s1), lc2(s2)
    (x, y, z) = get_lc1_lc2_features(parser_config, 1)
    word_features, pos_features, label_features = word_features + x, pos_features + y, label_features + z

    (x, y, z) = get_lc1_lc2_features(parser_config, 2)
    word_features, pos_features, label_features = word_features + x, pos_features + y, label_features + z


    # 4. Get word & pos & label features for rc1(s1), rc1(s2), rc2(s1), rc2(s2)
    (x, y, z) = get_rc1_rc2_features(parser_config, 1)
    word_features, pos_features, label_features = word_features + x, pos_features + y, label_features + z

    (x, y, z) = get_rc1_rc2_features(parser_config, 2)
    word_features, pos_features, label_features = word_features + x, pos_features + y, label_features + z


    # 5. Get word & pos & label features for lc1(lc1(s1)), lc1(lc1(s2)), rc1(rc1(s1)), rc1(rc1(s2))
    (x, y, z) = get_llc_rrc_features(parser_config, 1)
    word_features, pos_features, label_features = word_features + x, pos_features + y, label_features + z

    (x, y, z) = get_llc_rrc_features(parser_config, 2)
    word_features, pos_features, label_features = word_features + x, pos_features + y, label_features + z


    features = word_features + pos_features + label_features

    ######################################################################################
    # comment it out later
    # print(f"features: {features}")
    # print(f"len of features list: {len(features)}")
    # features: [43, 63, 53, 36, 71, 71, 54, 62, 49, 62, 71, 71, 34, 59, 71, 71, 71, 71, 23, 23, 23, 17, 32, 32, 21, 19, 21, 19, 32, 32, 16, 24, 32, 32, 32, 32, 3, 4, 3, 4, 15, 15, 14, 11, 15, 15, 15, 15]
    # len of features list: 48


    assert len(features) == 48
    return features


# Run the following cell as a sanity check for `generate_training_examples(..., feat_extract=extract_features)` (i.e., to make sure that you can generate training examples with the correct feature extraction function).

# In[38]:


if __name__ == '__main__':
    sanityCheck_generate_training_examples(generate_training_examples, extract_features, to_print='incorrect', do_raise=True)


# # Step 3: Dataset & Model [16 points]
# 
# Now we can go ahead and define our Pytorch `Dataset` class as well as our model.

# In[39]:


### DO NOT EDIT ###

import torch
import torch.nn as nn
import torch.nn.functional as F


# ## Instantiate Dataset
# 
# As in previous homeworks, we create a Pytorch `Dataset`, which we will use to feed our training data to the model. You do <b>not</b> need to edit this cell.

# In[40]:


### DO NOT EDIT ###

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data, vocab):
        self.X = np.array([d[0] for d in data])
        self.y = np.array([vocab.tran2id[d[1]] for d in data])
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


# ## <font color='red'>TODO</font>: Define Model [16 points]
# 
# Here you will write the `__init(...)__` and `forward(...)` methods of a feed-forward network, each of which is worth <b>8 points</b>. Your network should have an embedding layer, a single hidden layer, and an output layer. The `forward(...)` method will take in the features you have extracted from the parser configuration and predict the next parser action.

# In[41]:


class Model(nn.Module):
    def __init__(self, num_embeddings, embed_size, n_features, hidden_size, n_classes, dropout_prob):
        '''
        Initialize the weights of feed-forward neural network.
        Args:
            num_embeddings: Number of embedding vectors in embedding layer (int)
            embed_size: Size of the embedding vectors in embedding layer (int)
            n_features: Number of features in the input to the model (int)
            hidden_size: Hidden size (int)
            n_classes: Number of classes in output (int)
            dropout_prob: Probability of dropout (float)
        '''
        super(Model, self).__init__()

        ### TODO ###
        # Initialize embedding layer 
        # Initialize a linear layer that maps the (concatenated) embeddings to a single vector of size hidden_size
        # Create a dropout layer with dropout_prob
        # Initialize a linear layer that maps the hidden vector to the number of output classes
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"num_embeddings: {num_embeddings}, embed_size: {embed_size}")
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_size).to(device)

         # Obtain embedding vectors for your input
         #            - Output size: [batch_size, n_features * embed_size]
        self.linear_layer = nn.Linear(embed_size*n_features, hidden_size).to(device)

        self.dropout_layer = (nn.Dropout(dropout_prob)).to(device)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, n_classes).to(device)

        # pass

    def forward(self, x):
        '''
        This function predicts the next parser action, given the features extracted from the current parser state.
        Inputs:
             x: input features, [batch_size, n_features]
        Returns:
            logits: [batch_size, n_classes]
        Pseudocode:
            (1) Obtain embedding vectors for your input
                    - Output size: [batch_size, n_features * embed_size]
            (2) Pass the result through the first linear layer and apply ReLU activation
            (3) Apply dropout
            (4) Pass the result through the final linear layer and return its output (do NOT call softmax!)
                    - Output size: [batch_size, n_classes]
        '''

        ### TODO ###
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        # Shape of x: torch.Size([1, 32]) == [batch_size, n_features]
        # print('Content of x:', x)
        batch_size, n_features = x.shape
        # print('Shape of x input features, [batch_size, n_features]:', x.shape, '\n')
        # print('Type of x:', x.dtype, '\n')

        x_int64 = x.type(torch.int64)
        # print('Content of embedding:', x_int64)
        # print('Shape of embedding:', x_int64.shape, '\n')
        # print('Type of embedding:', x_int64.dtype, '\n')


        # word_embedding = (self.embedding(x)).to(device)
        # print('Content of embedding:', x_int64)
        # print('Shape of embedding:', x_int64.shape, '\n')
        # print('Type of embedding:', x_int64.dtype, '\n')


        ####################################################################
        # Obtain word embedding
        #  Obtain embedding vectors for your input
        #             - Output size: [batch_size, n_features * embed_size]
        word_embedding = (self.embedding(x_int64)).to(device) #[1, 32, 8]
        word_embedding_squeezed = word_embedding.squeeze(0)
        embed_size = word_embedding_squeezed.shape[-1]



        # print('Content of word_embedding:', word_embedding)
        # print('Shape of word_embedding:', word_embedding.shape, '\n')
        # print('Type of word_embedding:', word_embedding.dtype, '\n')
        # Shape of word_embedding: torch.Size([1, 32, 8])
        # word_embedding_squeezed = word_embedding.squeeze(0)  #[32, 8] but should be [1, 32*8] = [1, 256] = [batch size, num of features*embed size]
        word_embedding_reshaped = word_embedding.view(batch_size, n_features*embed_size)

        relu = self.relu(self.linear_layer(word_embedding_reshaped))

        dropout = self.dropout_layer(relu)

        result_of_output_layer = self.output_layer(dropout)
        # print('Content of result_of_output_layer:', result_of_output_layer)
        # print('Shape of result_of_output_layer:', result_of_output_layer.shape, '\n')
        # print('Type of result_of_output_layer:', result_of_output_layer.dtype, '\n')

         # result_of_output_layer = logits: [batch_size, n_classes]

        return result_of_output_layer


# The code below runs a sanity check for your model class. The tests are similar to the hidden ones in Gradescope. However, note that passing the sanity check does <b>not</b> guarantee that you will pass the autograder; it is intended to help you debug.

# In[42]:


### DO NOT EDIT ###

if __name__ == '__main__':
    # Test init
    inputs = [{'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 32, 'hidden_size': 32, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 3}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 2}, {'num_embeddings': 1000, 'embed_size': 32, 'n_features': 128, 'hidden_size': 256, 'dropout_prob': 0, 'n_classes': 3}]
    expected_outputs = [32482,32515,32482,32515,32482,32515,32482,32515,541058,541315,541058,541315,541058,541315,541058,541315,64866,64899,64866,64899,64866,64899,64866,64899,1081346,1081603,1081346,1081603,1081346,1081603,1081346,1081603]

    sanityCheckModel(inputs, Model, expected_outputs, "init")
    print()

    # Test forward
    forward_inputs = [{'num_embeddings': 1000, 'embed_size': 8, 'n_features': 32, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 32, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 32, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 32, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 64, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 64, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 64, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 64, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 64, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 64, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 64, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 64, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 32, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 32, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 32, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 32, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 64, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 64, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 64, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 8, 'n_features': 64, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 32, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 64, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 64, 'hidden_size': 100, 'dropout_prob': 0, 'n_classes': 80}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 64, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 40}, {'num_embeddings': 1000, 'embed_size': 16, 'n_features': 64, 'hidden_size': 200, 'dropout_prob': 0, 'n_classes': 80}]
    expected_outputs = [torch.Size([1, 40]), torch.Size([1, 80]), torch.Size([1, 40]), torch.Size([1, 80]), torch.Size([1, 40]), torch.Size([1, 80]), torch.Size([1, 40]), torch.Size([1, 80]), torch.Size([1, 40]), torch.Size([1, 80]), torch.Size([1, 40]), torch.Size([1, 80]), torch.Size([1, 40]), torch.Size([1, 80]), torch.Size([1, 40]), torch.Size([1, 80]), torch.Size([4, 40]), torch.Size([4, 80]), torch.Size([4, 40]), torch.Size([4, 80]), torch.Size([4, 40]), torch.Size([4, 80]), torch.Size([4, 40]), torch.Size([4, 80]), torch.Size([4, 40]), torch.Size([4, 80]), torch.Size([4, 40]), torch.Size([4, 80]), torch.Size([4, 40]), torch.Size([4, 80]), torch.Size([4, 40]), torch.Size([4, 80])]
    batch_sizes = [1] * (len(forward_inputs)//2) + [4] * (len(forward_inputs)//2)

    sanityCheckModel(forward_inputs, Model, expected_outputs, "forward", batch_sizes)


# # Step 4: Train Model
# 
# Finally, you are ready to train your model. We provide you with all the code you need to train it, so you do <b>not</b> need to edit any code in this section.

# In[43]:


### DO NOT EDIT ###

import math
from torch import optim
from tqdm.notebook import tqdm
import time


# First, we read in the training dataset, create training examples, extract features, and instantiate the Pytorch `Dataset`.

# In[44]:


### DO NOT EDIT ###

def prepare_data(train_name='train', test_name='test'):

    train_set, test_set = load_data()

    vocab = Vocabulary(train_set)
    vocab.printStats()
    print()

    train_set = vocab.buildSentences(train_set)
    test_set = vocab.buildSentences(test_set)

    train_examples = generate_all_training_examples(vocab, train_set, feat_extract=extract_features)

    return vocab, train_examples, test_set

if __name__== "__main__":
    vocab, train_examples, test_data = prepare_data()
    train_dataset = TrainDataset(train_examples, vocab)


# We will train the neural network using cross-entropy loss.

# In[45]:


### DO NOT EDIT ###

def train_model(model, vocab, train_data_loader, optimizer, n_epochs, device):
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        start = time.time()
        n_batch = 0
        total_loss = 0
        model.train()      
        for train_x, train_y in tqdm(train_data_loader):
            optimizer.zero_grad() 
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            logits = model(train_x)
            loss = loss_func(logits, train_y)
            loss.backward()
            optimizer.step()
            
            total_loss +=  loss.item()
            n_batch += 1
        
        print('Epoch:{:2d}/{}\t Loss: {:.4f} \t({:.2f}s)'.format(epoch + 1, n_epochs, total_loss / n_batch, time.time() - start))


# Next we instantiate the model and an <a href=https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>Adagrad</a> optimizer. As with other homeworks, you are free to change the hyperparameters, though you should not need to.

# In[46]:


### DO NOT EDIT ###

if __name__ == "__main__":
    # HYPERPARAMETERS - Feel free to change
    # BATCH_SIZE = 1024
    # LEARNING_RATE = 0.01
    # N_EPOCHS = 10
    # HIDDEN_SIZE = 300
    # DROPOUT_PROB = 0.1
    # EMBED_SIZE = 100
    # WEIGHT_DECAY = 1e-8

    BATCH_SIZE = 1024
    LEARNING_RATE = 0.03
    N_EPOCHS = 20
    HIDDEN_SIZE = 600
    DROPOUT_PROB = 0.1
    EMBED_SIZE = 200
    WEIGHT_DECAY = 1e-8

    N_EMBEDDINGS = vocab.n_tokens # Do not change!
    N_FEATURES = 48 # Do not change!
    N_CLASSES = vocab.n_trans # Do not change!
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    model = Model(N_EMBEDDINGS, EMBED_SIZE, N_FEATURES, HIDDEN_SIZE, N_CLASSES, DROPOUT_PROB).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# Run the cell below to train the model.

# In[47]:


### DO NOT EDIT ###

if __name__=='__main__':
    train_model(model, vocab, train_data_loader, optimizer, N_EPOCHS, device)


# # Step 5: Evaluate Model [28 points]
# 
# Now that you have a trained model, we can use it to parse unseen sentences from a test set.

# ## <font color='red'>TODO</font>: Select Best Legal Prediction [8 points]
# We will provide you with a function that takes a (trained) model and a batch of parser configurations, and returns the highest probability prediction for each configuration.
# 
# However, it is possible that the model will predict an illegal action. For example, the model may predict `S` (shift) when the buffer is empty, which is not a valid move. We certainly hope that the model will not do this, but we cannot guarantee this for an unseen sentence, and so have to account for the possibility at inference time.
# 
# Thus, you should return the highest probability <b>legal</b> action for each parser configuration. Your job is to write a function `select_best_legal_action(parser_configs, predictions, n_labels)` that does exactly this. You have some flexibility in how you choose to implement this. 
# 
# <font color='green'>Here are some <b>hints</b>:
# * `predictions` is of size `[N, 2*n_labels+1]`, where `N=len(parser_configs)`. It contains the (unnormalized) probabilities for each action as output by the `forward(...)` method of your model.
# * Consider a single row of `predictions`. The first `n_labels` indices `[0,...,n_labels-1]` correspond to the various `LA-label` actions. The second `n_labels` indices `[n_labels,...,2*n_labels-1]` correspond to the `RA-label` actions. The very last index `2*n_labels` corresponds to the `S` action.
# * For each row of `predictions`, you will want to determine which actions are legal. <b>You will need to think about how to tell which actions are legal based on a parser configuration.</b> Once you know this, we suggest building a bit map of size `2*n_labels+1`, where an index contains a `1` if its corersponding action is legal and a `0` otherwise.
# * You can then select the best action by either masking out the illegal actions, or by multiplying your bit map by a large number and adding it to the probabilities, so that the `argmax` operation can only return a legal index.
# * When thinking about which actions are legal, you do not need to worry about whether certain labels should be allowed or not. You just need to focus on when `LA`, `RA`, and `S` are allowed (regardless of whether a particular label makes sense on a particular `LA` or `RA` arc).</font>
# 
# This function is worth <b>8 points</b>, and there is no partial credit.

# In[81]:


def select_best_legal_action(parser_configs, predictions, n_labels):
    '''
    Returns the highest probability **legal** prediction for each parser configuration.
    Inputs:
        parser_configs: list of parser configurations of length N
        predictions: np.array of size [N, 2*n_labels + 1]
        n_labels: int, the number of labels in our model
    Returns:
        preds: np.array of length N, containing the indices of the highest probability legal action for each example
    '''
    # preds = np.argmax(predictions, axis = 1) # Change this! This selects the highest probability action, regardless of legality.

    ### TODO ###
    # print(f"parser_configs: {parser_configs}")
    # print(f"parser_configs list len: {len(parser_configs)}")
    parser_configs_len = len(parser_configs)

    # parser_configs: [<__main__.ParserConfiguration object at 0x7f54dcb02520>, <__main__.ParserConfiguration object at 0x7f54dcc49400>, <__main__.ParserConfiguration object at 0x7f54dea25f70>, <__main__.ParserConfiguration object at 0x7f54dcc65d00>, <__main__.ParserConfiguration object at 0x7f54dccb0460>, <__main__.ParserConfiguration object at 0x7f54dc80b5e0>,

    # print(f"predictions: {predictions}")
    # print(f"predictions shape: {predictions.shape}")
    # predictions shape: (1765, 95)    #1765 parser configs with 95 columns for each parser config
    # 95 = 2 * n_labels + 1 = 2*47+1 = 94 +1 = 95
    # n_labels: 47    # we have to predict the label?
    # print(f"n_labels: {n_labels}\n")



    # parser_configs list len: 1765
    # predictions: [[ -7.9638505   -4.9023805   -1.7691197  ...   5.3745494    0.02722144
    #    13.007633  ]
    #  [ -8.593049    -2.501113   -10.753521   ...  -2.5274448   -1.3838012
    #    15.562916  ]
    #  [ -7.4554424    0.36085862  -4.11383    ...   2.9703796    5.904849
    #    12.831154  ]
    #  ...
    #  [ -9.084528     0.5178829   -6.563924   ...  -2.7183692    3.8894663
    #    18.006874  ]
    #  [ -7.640521    -6.610285    -8.546304   ...  -3.785166    -3.4325747
    #    12.347177  ]
    #  [ -6.3907423   -4.6567173   -8.828304   ...  -8.035017    -0.10355127
    #    14.019371  ]]

    # class ParserConfiguration(object):
    #     def __init__(self, sentence, vocab):
    #         '''
    #         Inputs:
    #             sentence: list of Word objects
    #             vocab: Vocabulary object
    #         '''
    #
    #         self.vocab = vocab
    #
    #         assert sentence[0].word_id == self.vocab.WORD_ROOT
    #         self.stack = Stack([sentence[0]]) # Initialize stack with ROOT
    #         self.buffer = Buffer(sentence[1:]) # Initialize buffer with sentence
    #         self.dependencies = Dependencies(vocab)
    #
    #     def parse_step(self, transition):
    #         '''
    #         Update stack, buffer, and dependencies based on transition.
    #         Inputs:
    #             transition: str, "S", "LA-label", or "RA-label", where label is a valid label
    #         '''
    #         assert transition in self.vocab.tran2id
    #
    #         ### TODO ###
    #
    #         pass

    bitmap_np_array = np.zeros((parser_configs_len, 2*n_labels + 1))
    # print(f"bitmap_np_array shape before for loops: {bitmap_np_array.shape}")



    # parser_configs: list of parser configurations of length N = 1765
    for index, parser_config in enumerate(parser_configs):
        parser_config_probs = predictions[index]
        bitmap_1D_np_array = np.zeros((2*n_labels + 1))
        # print(f"probs of index {index} are: {parser_config_probs}")

        stack = parser_config.stack
        buffer = parser_config.buffer
        # dependencies = parser_config.dependencies
        # print(f"dependecies are: {dependencies}")

        # print(f"index is: {index}")
        # print(f"stack len is: {len(stack)}")
        # print(f"buffer len is: {len(buffer)}")

        # S
        if len(buffer) > 0 :
            # print(f"S is a valid action for this parser config")
            bitmap_1D_np_array[2*n_labels] = 1
            # bitmap_1D_np_array[:2*n_labels] = 0

        # RA
        # if (len(stack) >= 3) or (len(buffer) == 0 and len(stack) == 2):
        if len(stack) >= 2:
            # print(f"RA-label is a valid action for this parser config")
            # bitmap_1D_np_array[0:n_labels] = 0
            # bitmap_1D_np_array[n_labels:2*n_labels-1] = 1
            bitmap_1D_np_array[n_labels:2*n_labels] = 1
            # bitmap_1D_np_array[2*n_labels] = 0


        # LA
        # if (len(stack) >= 3) or (len(buffer) == 0 and len(stack) == 2):
        if len(stack) >= 3:
            # print(f"LA-label is a valid action for this parser config")
            # bitmap_1D_np_array[0:n_labels-1] = 1
            bitmap_1D_np_array[0:n_labels] = 1
            # bitmap_1D_np_array[n_labels:] = 0

        predictions[index] = predictions[index] + (bitmap_1D_np_array*100)

        # #############################################
        # # with 2D bitmap
        # # S
        # if len(buffer) > 0 :
        #     # print(f"S is a valid action for this parser config")
        #     bitmap_np_array[index, 2*n_labels] = 1
        #     # bitmap_np_array[index, :2*n_labels] = 0
        #
        # # RA
        # if (len(buffer) > 0 and len(stack) >= 2) or (len(buffer) == 0 and len(stack) == 2):
        # # if (len(stack) >= 2):
        #     # print(f"RA-label is a valid action for this parser config")
        #     # bitmap_np_array[index, 0:n_labels] = 0
        #     bitmap_np_array[index, n_labels:2*n_labels] = 1
        #     # bitmap_np_array[index, 2*n_labels] = 0
        #
        #
        # # LA
        # if (len(buffer) > 0 and len(stack) >= 3) or (len(buffer) == 0 and len(stack) == 2):
        # # if (len(stack) >= 3):
        #     # print(f"LA-label is a valid action for this parser config")
        #     bitmap_np_array[index, 0:n_labels] = 1
        #     # bitmap_np_array[index, n_labels:] = 0

    # print(f"bitmap_np_array: {bitmap_np_array}")
    # print(f"bitmap_np_array.shape after for loop: {bitmap_np_array.shape}")

    # parser_config_probs = bitmap_1D_np_array
    # valid_action_probs = predictions + (bitmap_np_array*1000)

    valid_action_probs = predictions
    # print(f"valid_action_probs: {valid_action_probs}")
    # print(f"valid_action_probs.shape: {valid_action_probs.shape}")

    argmax_for_index = np.argmax(valid_action_probs, axis = 1)
    # print(f"argmax_for_index: {argmax_for_index}")
    # print(f"argmax_for_index.shape: {argmax_for_index.shape}\n\n")
    # print(f"######################################## select_best_legal_action FUNCTION COMPLETED #################################")




        # if len
        #     legal_action =

        # The first n_labels indices [0,...,n_labels-1] correspond to the various LA-label actions
        # The second n_labels indices [n_labels,...,2n_labels-1] correspond to the RA-label actions.
        # The very last index 2n_labels corresponds to the S action.

        # for each row of predictions, we need to find 1 best possible action among LA-label actions [0,...,n_labels-1] , RA-label actions and Shift action
        # each parser config is the config after an action, every parser config corresponds to a row of preds probabilities. We just need to find the legal action.


        # BITMAP
        # Lets say for Row 1  , LA action is legal then all RA-labels and shift label t should be 0 in bitmap and we select the LA action label with max probability
        # for each row of the array you will find one best action
        # For LA-label if the action is legal, you will set your bit_map to bit_map[0:n_labels-1] = 1. Once you’re done with the entire array, then you can multiply the bit_map array to the individual prediction row in the  predictions array. Then select the max from the predictions array once you’re done masking the entire predictions array.
        # RA (set  [n_labels:2*n_labels-1] )and S (set [2*n_labels])
        # just multiply bitmap by the pred row and take the argmax of all predictions over axis=1


    return argmax_for_index


# Now we provide you with a function that takes a (trained) model and makes the best legal prediction for a batch of parser configurations. You do <b>not</b> need to edit this cell.

# In[82]:


### DO NOT EDIT ###

def predict(model, vocab, parser_configs):
    '''
    Predicts the next transition for each ParserConfiguration in the batch (`parsers`).
    '''
    model_device = next(model.parameters()).device
    
    x = np.array([extract_features(p) for p in parser_configs])
    x = torch.from_numpy(x).long().to(model_device)
    
    with torch.no_grad():
        pred = model(x)

    pred = pred.detach().cpu().numpy()
    pred = select_best_legal_action(parser_configs, pred, vocab.n_labels)

    #########################################
    # print(f"pred is: {pred}")
    actions = [vocab.id2tran[p] for p in pred]
    return actions


# ## Test Set Attachment Score [20 points]
# 
# The following functions use your model to parse all sentences in the test set, and compute the attachment score. The <b>unlabeled attachment score</b> is the percentage of arcs in the test set for which your model gets the head correct. The <b>labeled attachment score</b> is the percentage of arcs for which your model gets <em>both</em> the head and the label correct. Thus, attachment score is a number between 0 and 100, and a higher score is better.
# 
# You do <b>not</b> need to edit this cell.

# In[88]:


### DO NOT EDIT ###

def run_inference(sentences, model, vocab, batch_size=2000):
    '''
    Infers the dependency parse for each sentence given a trained model.
    '''
    N = len(sentences)

    # Initialize parser configs
    parser_configs = [None] * N
    for i in range(N):
        sent = sentences[i]
        parser_config = ParserConfiguration(sent, vocab)
        parser_configs[i] = parser_config

    parses_completed = [False] * N # Indicates whether a given parse is completed
    
    while sum(parses_completed) != N:

        # Get batch along with indices
        batch_idxes = []
        for idx in range(N):
            if not parses_completed[idx]: batch_idxes.append(idx)
            if len(batch_idxes) == batch_size: break
        batch = [parser_configs[idx] for idx in batch_idxes]

        # Make prediction, run a parse step, and check for completion
        transitions = predict(model, vocab, batch)
        for idx, parser, transition in zip(batch_idxes, batch, transitions):
            parser.parse_step(transition)
            if not parser.buffer.buffer and len(parser.stack.stack) == 1:
                parses_completed[idx] = True
    
    return [parser.dependencies for parser in parser_configs]

def transform_to_head_label(dependencies):
    head = [-1] * len(dependencies.arcs)
    label = [-1] * len(dependencies.arcs)
    for dep in dependencies.arcs:
        head[dep.dependent.idx-1] = dep.head.idx
        label[dep.dependent.idx-1] = dep.label_id  
    return head, label


def evaluate(model, vocab, dataset, eval_batch_size=5000):
    model.eval()
    sentences = [x[0] for x in dataset]
    gold_dependencies = [x[1] for x in dataset]
    pred_dependencies = run_inference(sentences, model, vocab, eval_batch_size)

    # print(f"pred_dependencies: {pred_dependencies}")
    # print(f"pred_dependencies len: {len(pred_dependencies)}")
    
    UAS, LAS = 0.0, 0.0

    all_tokens = 0
    
    for i in range(len(gold_dependencies)):
        assert len(gold_dependencies[i].arcs) == len(pred_dependencies[i].arcs)
        
        # Get gold answers
        gold_head, gold_label = transform_to_head_label(gold_dependencies[i])
        
        # Get predictions
        pred_head, pred_label = transform_to_head_label(pred_dependencies[i])

        # print(f"pred_head: {pred_head}")
        # print(f"pred_head len: {len(pred_head)}")
        #
        # print(f"pred_label: {pred_label}")
        # print(f"pred_label len: {len(pred_label)}")

        
        assert len(gold_head) == len(pred_head) and len(gold_label) == len(pred_label)
        assert -1 not in gold_head + gold_label + pred_head + pred_label

        for pred_h, gold_h, pred_l, gold_l  in zip(pred_head, gold_head, pred_label, gold_label):
            UAS += (1 if pred_h == gold_h else 0)
            LAS += (1 if pred_h == gold_h and pred_l == gold_l else 0)
            all_tokens += 1
    return UAS / all_tokens * 100, LAS / all_tokens * 100, pred_dependencies


# Run the following cell to calculate your attachment scores. You must achieve a <b>labeled attachment score</b> of <b>≥ 80%</b> for full credit. Bear in mind that Gradescope uses a different (hidden) test set, so results may be slightly different.

# In[89]:


### DO NOT EDIT ###

if __name__=="__main__":
    UAS, LAS, test_predictions = evaluate(model, vocab, test_data)
    print("Test Set Unlabeled Attachment Score:", UAS)
    print("Test Set Labeled Attachment Score:", LAS)


# ## Qualitative Analysis
# 
# This section allows you to analyze your model qualitatively to get a feel for the strengths and shortcomings of the model. You do <b>not</b> need to code anything in this section.
# 
# Run the following cells to print some example sentences from the test set. For each sentence, it will display the gold (correct) dependency tree, the dependency tree predicted by your model, and a diagnostic of the gold tree. The diagnostic tree annotates the edges of the gold tree as follows:
# * ✓: Edges for which you predicted both the <b>correct head & label</b>
# * ⍻: Edges for which you predicted the <b>correct head but incorrect label</b>
# * ×: Edges that you do not have in your tree (i.e., you predicted the <b>incorrect head<b>).

# In[90]:


### DO NOT EDIT ###

def diagnose(sentence, gold, pred):
    word, pos = [x.word for x in sentence], [x.pos for x in sentence]
    gold_head, gold_label = transform_to_head_label(gold)
    pred_head, pred_label = transform_to_head_label(pred)
    gold_label = [gold.vocab.id2tok[x][4:] for x in gold_label]
    pred_label = [gold.vocab.id2tok[x][4:] for x in pred_label]

    diff = ["✓"] * len(pred_head)
    for i in range(len(pred_head)):
        if gold_head[i]!=pred_head[i]: diff[i] = "×"
        elif gold_label[i]!=pred_label[i]: diff[i] = "⍻"
    unlabeled_score=sum([x in {"✓","⍻"}  for x in diff]) / len(diff)
    score=sum([x == "✓" for x in diff]) / len(diff)

    print("Your dependency tree:")
    sent = {"word": word[1:], "pos": pos[1:], "label": pred_label, "head": pred_head}
    display_sentence(sent)
    
    print("Gold dependency tree:")
    sent["label"], sent["head"] = gold_label, gold_head
    display_sentence(sent)
    
    print("Diagnostic of the gold tree:")
    sent["label"], sent["head"] = diff, gold_head
    display_sentence(sent)
    print('Unlabeled Attachment Score for this sentence:', unlabeled_score*100)
    print('Labeled Attachment Score for this sentence:', score*100, '\n')

def diagnose_sentences(idxes, data, preds, min_len, max_len, num_to_print=5):
    print('-'*100, '\n')
    for i in random.sample(list(filter(lambda x: len(data[x][0]) >= min_len and len(data[x][0]) <= max_len, idxes)), num_to_print):
        diagnose(data[i][0], data[i][1], preds[i])
        print('-'*100, '\n')


# In[91]:


if __name__== '__main__':
    MIN_SENT_LEN, MAX_SENT_LEN = 8, 17 # You can change this if you're interested in seeing shorter/longer sentences
    idxes=list(range(len(test_data))) # Sample from all sentences
    diagnose_sentences(idxes, test_data, test_predictions, MIN_SENT_LEN, MAX_SENT_LEN, num_to_print=5)


# # What to Submit
# 
# To submit the assignment, download this notebook as a <TT>.py</TT> file. You can do this by going to <TT>File > Download > Download .py</TT>. Then rename it to `hwk4.py`. <b>Do not try to submit it as a <TT>.ipynb</TT> file!</b>
# 
# You will also need to save your trained `model`. You can run the cell below to do this. After you save the files to your Google Drive, you need to manually download the file to your computer, and then submit them to the autograder.
# 
# You will submit the following files to the autograder:
# 1.   `hwk4.py`, the download of this notebook as a `.py` file (**not** a `.ipynb` file)
# 1.   `model.pt`, the saved version of your `model`

# In[92]:


### DO NOT EDIT ###

import pickle

if __name__=='__main__':
    # from google.colab import drive
    # drive.mount('/content/drive')
    print()
    print("Saving model....") 
    # torch.save(model, 'drive/My Drive/model.pt')
    torch.save(model, './saved_models/model.pt')
    print("Saved!")


# In[92]:





# In[ ]:




