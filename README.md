# GEC
Grammatical Error Correction application using <https://streamlit.io>

Python libraries needed:
* [streamlit](https://streamlit.io) `$> pip install streamlit`
* [pyonmttok](https://github.com/OpenNMT/Tokenizer) `$> pip install pyonmttok`
* [ctranslate2](https://github.com/OpenNMT/CTranslate2) `$> pip install ctranslate2`
* [fasttext](https://fasttext.cc) `$> pip install fasttext`

To run the app: 

`$> streamlit run correction_app.py `

and open a web browser: <http://localhost:8501/>

## Install correction models

Assuming BASE=/directory/where/your/correction_app.py/is/located

### Prepare the LID model

* Download the LID model

`(mkdir -p $BASE/LID && cd $BASE/LID && wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz)`

### Prepare an Enc-Dec model

* Model directory must be EncDec_\[language\]

`mkdir -p $BASE/EncDec_en`

* The directory must contain:
  * The Transformer enc-dec model (ctranslate2 format), it contains:
    * __model.bin__
    * __source_vocabulary.txt__
    * __target_vocabulary.txt__
  * The same tokenization config file that you used for training:
    * __tok_conf__
  * The BPE file used in tok_conf file (if any)
    * __bpe.incorr__ (use the name indicated in the tok_conf file)

Example of tok_conf file:
```
mode: aggressive
joiner_annotate: True
preserve_segmented_tokens: True
segment_numbers: True
segment_alphabet_change: True
bpe_model_path: EncDec_en/bpe.incorr
```
Convert OpenNMT/Fairseq models to ctranslate2 format using:
```
ct2-opennmt-py-converter
ct2-opennmt-tf-converter
ct2-fairseq-converter
```

### Prepare a SriLM model


### Prepare a GECToR model
