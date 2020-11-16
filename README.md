# Real-time-sentiment-analysis-of-tweets-using-Seq2Seq-and-Trasformer-model
Comparison of custom n-to-1 seq2seq with attention mechanism and BERT transformer model in sentiment analysis of tweets coupled with open-source web-app for opinion-mining of real-time tweets. Pls refer to fyp_presentation in thesis_docs for more insights. 

BRIEF OVERVIEW:

CUSTOM-BUILT N-TO-1 SEQ2SEQ MODEL WITH ATTENTION: 
GloVe embeddings.
Encoder layer- Bidirectional GRU, embedding.

Attention layer- Hierarchical attention network (on colab)

The attention layer consists of feeding vectors of hidden states into a learning function which is a feed-forward neural network that produces a probability vector. This is used along with the weighted average of the hidden layer vectors to compute the context vector for this attention-based model.

Decoder layer- Feed-forward NN.


TRANSFORMER MODEL:
1. Simple BERT.
2. BERT trained partially on hashtag specific tweets that are manually annotated.


WEB APP:
Perform real-time opinion-mining on tweets pertinent to desired hashtag. 

GOLD STANDARD CORPUS:
Peformed by collecting manually annotated tweets from 5 human experts and using Fleiss' Kappa to find out inter-rater aggrement. 
Done by distributing tweets to evaluators by means of custom-built Google Forms.
