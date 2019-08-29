#!/usr/bin/env bash
# Create data directory
mkdir data
# Download & extract the zipped dataset from URL
curl http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz --output aclImdb_v1.tar.gz
# Unzip the file
tar -xzvf aclImdb_v1.tar.gz --directory data
# Remove the archive
rm aclImdb_v1.tar.gz