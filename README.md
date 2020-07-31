# upsilon_bleu

This is the codes used in [**υBLEU: Uncertainty-Aware Automatic Evaluation Method for Open-Domain Dialogue Systems**](https://www.aclweb.org/anthology/2020.acl-srw.27/) accepted by ACL-SRW 2020.



## Construction

This GitHub page includes 

- "data_arrangement" folder:

  Explain the data used in υBLEU and show the construction of them.

-  "src" folder: 

  Source codes of υBLEU and [ΔBLEU](https://www.aclweb.org/anthology/P15-2073) used in my work. The paths described in these codes are designed for the data in "data_arrangement" folder. All codes are written by Python3.
  
  <!-- [RUBER](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16179/15752) will also be deployed -->



## Usage

υBLEU is constructed by three parts.

- src/bleu_type_metrics/collectResponse.py

  Augment reference responses for υBLEU or ΔBLEU.

- src/bleu_type_metrics/nn-rater/*

  Rate the augmented reference responses by a classifier, named NN-rater.

- src/bleu_type_metrics/multiBleu.py

  Score generated responses by ΔBLEU.



Following shows the implementation in our paper.

##### collectResponse.py

###### Preparation:

- prepare the conversation data (the pairs of utterance and response) to test evaluation metrics
  - the data of "data_arrangement/test"

- make [SentencePiece](https://www.aclweb.org/anthology/D18-2012/) ([GitHub](https://github.com/google/sentencepiece)) model ("*.model" on the default setting)

- prepare the tokenized conversation data which responses are retrieved for reference response
  - the data of "data_arrangement/corpus" 

- make [Glove](https://www.aclweb.org/anthology/D14-1162/) vector (using the tool in [here](https://nlp.stanford.edu/projects/glove/))



###### Command Line:

To collect responses by our  method,

```
python src/bleu_type_metrics/collectResponse.py data_arrangement/test data_arrangement/corpus/tokenized/sentencepiece -tokenize sentencepiece -train <path to sentencepiece model (*.model)> -corpus_original data_arrangement/corpus -vector <path to globe vector> -method embedding -target utr -add_utr -add_rpl
```

To collect responses by the method written in ΔBLEU, 

(Note: This command uses [MeCab](https://taku910.github.io/mecab/) to tokenize Japanese sentences)

```
python src/bleu_type_metrics/collectResponse.py data_arrangement/test data_arrangement/corpus/tokenized/mecab -tokenize mecab -corpus_original data_arrangement/corpus -method bm25 -target dialog -add_utr -add_rpl
```

###### Note:

If you do not use the option "-write_path" (or "-wp"), new folder is created in the test data directory ("data_arrangement/test/pseudos" is created in the default settings). This new directory includes the files that save reference responses for each test example, retrieved from corpus. 



See the description of "collectResponse.py" by executing the following command when you see the details.

```
python src/bleu_type_metrics/collectResponse.py -h
```



##### nn-rater/*

###### Preparation:

- prepare the training data for the NN-rater
  
  - the data of "data_arrangement/train"
  
- execute "collectResponse.py" for "nn-rate/test.py"

  

###### Command Line:

To train the model,

```
python3 src/bleu_type_metrics/nn-rater/train.py data_arrangement/train -save <the folder to save the model of NN-rater>
```

To test the model (, or add the probability to the reference responses),

```
python3 src/bleu_type_metrics/nn-rater/test.py data_arrangement/test <path to collected responses> <path to model file> -tokenize sentencepiece -train <path to sentencepiece model (*.model)>
```



##### multiBleu.py

###### Preparation:

- prepare the generated responses for the test samples and human annotation score for them.
  - the data of "data_arrangement/test/gen.rep" and "data_arrangement/test/human_score"
- execute "nn-rater/test.py" to qualify the collected reference responses



###### Command Line:

To calculate correlations with human judgment,

(Note: This command uses [MeCab](https://taku910.github.io/mecab/) to tokenize Japanese sentences)

```
python3 src/bleu_type_metrics/multiBleu.py data_arrangement/test <path to collected responses> -tokenize mecab
```



## Reference

```
@inproceedings{yuma-etal-2020-ubleu,
    title = "u{BLEU}: Uncertainty-Aware Automatic Evaluation Method for Open-Domain Dialogue Systems",
    author = "Yuma, Tsuta  and
      Yoshinaga, Naoki  and
      Toyoda, Masashi",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-srw.27",
    pages = "199--206"
}
```
