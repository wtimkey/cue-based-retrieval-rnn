# A Neural Model of Cue-Based Retrieval in Sentence Processing


## Training models from scratch
The code in this repository built upon [neural-complexity](https://github.com/vansky/neural-complexity) and [colorlessgreenRNNs](https://github.com/facebookresearch/colorlessgreenRNNs). Currently, data must first be preprocessed into a an hdf5 file using the `gen_dataset.py` script. This file takes a list of pre-tokenized, CCG supertagged sentences, and prepares it for use with our model as an HDF5 file. You can specify the desired vocabulary size, whether to collapse numerals to a single `<num>` token, and whether to lowercase the input. 

I will add an option to use BPE tokenization at a later date. Note that the data handling components of the model have been significantly reworked for speed and readability since the models that were trained for this paper. The pretrained models from our paper can be found [here](). The original data handling code is available upon request.

Our raw training dataset is available here. You can generate a model-ready dataset with the following:

`python gen_dataset.py --train_input_fname data/wikitext103_ccgtagged.train --valid_input_fname data/wikitext103_ccgtagged.valid --test_input_fname data/wikitext103_ccgtagged.test --lower --ccg_supertags --vocab_fname ./data/vocab.txt --aux_vocab_fname ./aux_vocab.txt`

The script will generate the HDF5 files in the same location as the input files. To train the CBR-RNN models (with a CCG-supertagging auxilary objective), you can use the following:

`python main.py --model CBRRNN --data_dir ./data/ --objective lm --aux_objective --vocab_file ./data/vocab.txt --aux_vocab_file aux_vocab.txt --trainfname wikitext103_ccgtagged.train.hdf5 --validfname wikitext103_ccgtagged.valid.hdf5 --tied --model_file cbrrnn_test_model.pt`

Consult `main.py` for the full set of possible arguments to the model. You can change the embedding and hidden layer sizes, whether weights are tied, dropout, batch size etc.

## Replication of Timkey and Linzen (2023)
Data for replicating results from Timkey and Linzen (2023) "A Language Model with Limited Memory Capacity Captures Interference in Human Sentence Processing" can be found in the folder `emnlp_analysis_data`. I have compiled our analysis into a jupyter notebook `Analysis.ipynb`. Run this notebook first. This will generate the files needed to run the statistical analysis and generate plots in R, using the Rmd file `emnlp_analysis_data/emnlp_analysis.Rmd`. If you just want to look any any part of the analysis, the pre-computed results files are available in the analysis folder. Send an email or open an issue if you run into any difficulties!
