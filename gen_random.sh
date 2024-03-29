#!/bin/bash

# Script to compute and save random embedding vectors for a given source vocab.  
#srun -J rando --mem 30000 -c 20 -w osmium python3 rand_vecs.py ../embeddings/top_10000_emb.txt

# Glove
#srun -J rando --mem 50000 -c 3 -w adamantium python3 rand_vecs.py ../embeddings/glove.840B.300d.txt Glove


#srun -J rando --mem 50000 -c 3 -w adamantium python3 rand_vecs.py ../embeddings/GoogleNews-vectors-negative300.bin Word2Vec
srun -J genrand --mem 10000 -c 10 -w adamantium python3 rand_vecs.py /data/data26/scratch/user/pretrained-embeddings/GoogleNews-vectors-negative300.bin Word2Vec
#srun -J genrand --mem 10000 -c 10 -w osmium python3 rand_vecs.py /data/data26/scratch/user/pretrained-embeddings/glove.840B.300d.word2vec_clean.bin Word2Vec
#srun -J genrand --mem 10000 -c 10 -w osmium python3 rand_vecs.py /data/data26/scratch/user/pretrained-embeddings/wiki-news-300d-1M-subword.bin Word2Vec
#srun -J genrand --mem 10000 -c 10 -w osmium python3 rand_vecs.py /data/data26/scratch/user/pretrained-embeddings/word2vec-text8-vectors.bin Word2Vec
