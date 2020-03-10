import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
from FilesIO import save_JSON_data
from enum import Enum
from numpy.linalg import norm


class VectorizationType(Enum):
    first_sentence = 1
    summary = 2
    full_text = 3


class TextVectorizer:
    def __init__(self, data, vectorized_data_file_name, vectorization_types, should_vectorize, should_normalize=True,
                 verbose = True):
        if verbose:
            print("TextVectorizer: starts")
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained('bert-base-uncased')
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        self.data = data
        self.vectorize = None
        if should_vectorize:
            for vec_type in vectorization_types:
                self.vectorize = self.enum_to_vectorization_function(vec_type)
                self.vectorize_data(vec_type, should_normalize)
                print("TextVectorizer: vectorization:" + str(vec_type) + " completed.")
                print("TextVectorizer: saving vectorized data to JSON file...")
                save_JSON_data(self.data, vectorized_data_file_name + ".json", beautify=True)
        if verbose:
            print("TextVectorizer: finished")

    def vectorize_data(self, vectorization_type, should_normalize):
        print("TextVectorizer: vectorizing...")
        for article in self.data.articles.values():
            vector = self.vectorize(article)
            print("TextVectorizer: vectorized: " + article.title)
            if should_normalize and vector is not None:
                vector = self.normalize_vector(vector)
            article.vectors[str(vectorization_type)] = vector

    def enum_to_vectorization_function(self, enum):
        switcher = {
            VectorizationType.first_sentence : self.first_sentence_vectorization,
            VectorizationType.summary : self.summary_vectorization,
            VectorizationType.full_text : self.full_text_vectorization
        }
        # maps the enum to the vectorization function, if not found returns None by default
        return switcher.get(enum, None)

    def normalize_vector(self, vector):
        norma = norm(vector)
        normalized_vector = torch.div(vector, norma)
        return normalized_vector

    def get_sentence_encoded_layers_vector(self, sentence):
        marked_sentence = "[CLS] " + sentence + " [SEP]"
        tokenized_sentence = self.tokenizer.tokenize(marked_sentence)
        # Bert does not support tokenized sentences with length greater than 512 tokens
        if len(tokenized_sentence) > 512:
            return None
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        segments_ids = [1] * len(tokenized_sentence)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor,segments_tensors)
        return encoded_layers

    def vectorize_sentence(self,sentence):
        encoded_layers = self.get_sentence_encoded_layers_vector(sentence)
        # encoded_layers will be None is sentence is greater than 512 words in length
        if encoded_layers is None:
            return None
        # remove the vectors of the [CLS] & [SEP] tokens: (narrow(dimension,start,length))
        encoded_layers = [tensor.narrow(1,1,tensor.shape[1]-2) for tensor in encoded_layers]
        sentence_vector = torch.mean(encoded_layers[10],1)
        return sentence_vector

    def get_sentence_word_vectors(self, sentence):
        encoded_layers = self.get_sentence_encoded_layers_vector(sentence)
        # Convert the hidden state embeddings into single token vectors
        # Holds the list of 12 layer embeddings for each token
        # Will have the shape: [# tokens, # layers, # features]
        token_embeddings = []
        # for each token:
        for token_i in range(len(encoded_layers[0][0])):
            # this will hold 12 layers of hidden states for each token:
            hidden_layers = []
            # for each of the 12 layers:
            for layer_i in range(len(encoded_layers)):
                vec = encoded_layers[layer_i][0][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
        summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in
                                token_embeddings]
        return summed_last_4_layers

    def __vectorize_sentences_array(self, sentences):
        if sentences is None or len(sentences) == 0:
            return None
        vectors = []
        for sentence in sentences:
            sentence_vector = self.vectorize_sentence(sentence)
            if sentence_vector is not None:
                vectors.append(sentence_vector)
        if len(vectors) == 0:
            return None
        return torch.mean(torch.stack(vectors),0)

    def first_sentence_vectorization(self, article):
        # sent_tokenize is an NLTK function which breaks text into an array of sentences
        sentences = sent_tokenize(article.summary)
        first_sentence = sentences[0]
        return self.vectorize_sentence(first_sentence)

    def summary_vectorization(self, article):
        sentences = sent_tokenize(article.summary)
        return self.__vectorize_sentences_array(sentences)

    def full_text_vectorization(self, article):
        full_text = article.summary
        if full_text is None:
            full_text = ""
        for section in article.sections:
            full_text += section[1]
        sentences = sent_tokenize(full_text)
        return self.__vectorize_sentences_array(sentences)
