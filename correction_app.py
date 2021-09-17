import sys
import time
import os
from datetime import datetime
from difflib import SequenceMatcher
import yaml
import six
import ctranslate2
import pyonmttok
import fasttext
import streamlit as st

now = datetime.now().strftime("[%d/%m/%Y-%H:%M:%S] ")
newline = '  \n' # first two spaces are needed
fasttext.FastText.eprint = lambda x: None #disable fasttext warnings
showWarningOnDirectExecution = False #disable streamlit warnings
device="cpu"

####################################################################################
####################################################################################
####################################################################################

def mark_diffs(lsrc, lhyp):
    #lsrc / lhyp are lists of strings where each string is a sentence
    #output is a string with all sentences
    for l in range(len(lsrc)):
        src = lsrc[l].split()
        hyp = lhyp[l].split()
        for tag, i, j, u, v in SequenceMatcher(None, src, hyp).get_opcodes():
            if tag in ('delete', 'insert', 'replace'):
                for k in range(u,v):
                    hyp[k] = '**' + hyp[k] + '**' ### word lhyp[l][k] is marked with bold face
        lhyp[l] = ' '.join(hyp)
    return newline.join(lhyp)

class tokenizer():
    def __init__(self, ftok):
        with open(ftok) as yamlfile: 
            opts = yaml.load(yamlfile, Loader=yaml.FullLoader)
        local_args = {}
        for k, v in six.iteritems(opts):
            if isinstance(v, six.string_types):
                local_args[k] = v.encode('utf-8')
            else:
                local_args[k] = v
        mode = local_args['mode']
        del local_args['mode']
        self.T = pyonmttok.Tokenizer(mode, **local_args)

    def tokenize(self, lsrc):
        #input is a list of strings where each string is a sentence
        #output is a list of list of strings where each string is a token
        lsrc_ltok = []
        for l in range(len(lsrc)):
            ltok, _ = self.T.tokenize(lsrc[l])
            lsrc_ltok.append(ltok)
        return lsrc_ltok

    def detokenize(self, lhyp_ltok):
        #input is a list of list of strings where each string is a token
        #output is a list of strings where each string is a sentence
        lhyp_detok = []
        for l in range(len(lhyp_ltok)):
            hyp = self.T.detokenize(lhyp_ltok[l])
            lhyp_detok.append(hyp)
        return lhyp_detok

####################################################################################
####################################################################################
####################################################################################

class EncDec():
    def __init__(self, lang, device='cpu'):
        self.mydir = 'EncDec_{}'.format(lang)
        self.tokenizer = tokenizer('{}/tok_conf'.format(self.mydir)) if os.path.isfile(self.mydir+'/tok_conf') else None
        self.translator = ctranslate2.Translator(self.mydir, device) if os.path.isfile(self.mydir+'/model.bin') else None
    
    def __len__(self):
        if self.tokenizer is None or self.translator is None:
            return 0
        return 1

    def correct(self, lsrc, now):
        lsrc_ltok = self.tokenizer.tokenize(lsrc)
        hyps = self.translator.translate_batch(lsrc_ltok)
        hyps = [hyp[0]["tokens"] for hyp in hyps]
        lhyps_detok = self.tokenizer.detokenize(hyps)
        sys.stdout.write(now+'{}:\t{}\n'.format(self.mydir,lhyps_detok))
        return lhyps_detok

class SriLM():
    def __init__(self, lang):
        self.mydir = 'SriLM_{}'.format(lang)
        self.tokenizer = tokenizer('{}/tok_conf'.format(self.mydir)) if os.path.isfile(self.mydir+'/tok_conf')  else None
        #sys.stdout.write('Built SriLM from: {}\n'.format(self.mydir))
    
    def __len__(self):
        if self.tokenizer is None:
            return 0
        return 1

    #@st.cache(suppress_st_warning=True)
    def correct(self, lsrc, now):
        lsrc_ltok = self.tokenizer.tokenize(lsrc)
        ### correct
        lhyps_ltok = list(lsrc_ltok)
        lhyps_detok = self.tokenizer.detokenize(lhyps_ltok)
        sys.stdout.write(now+'{}:\t{}\n'.format(self.mydir,lhyps_detok))
        return lhyps_detok

class GECToR():
    def __init__(self, lang, device='cpu'):
        self.mydir = 'GECToR_{}'.format(lang)
        self.tokenizer = tokenizer('{}/tok_conf'.format(self.mydir)) if os.path.isfile(self.mydir+'/tok_conf')  else None
        #sys.stdout.write('Built GECToR from: {}\n'.format(self.mydir))
    
    def __len__(self):
        if self.tokenizer is None:
            return 0
        return 1

    #@st.cache(suppress_st_warning=True)
    def correct(self, lsrc, now):
        lsrc_ltok = self.tokenizer.tokenize(lsrc)
        ### correct
        lhyps_ltok = list(lsrc_ltok)
        lhyps_detok = self.tokenizer.detokenize(lhyps_ltok)
        sys.stdout.write(now+'{}:\t{}\n'.format(self.mydir,lhyps_detok))
        return lhyps_detok
    
class Lid():
    def __init__(self):
        lid_model = "LID/lid.176.ftz"
        self.lid = fasttext.load_model(lid_model)
        #sys.stdout.write('Built Lid from: {}\n'.format(lid_model))

    #@st.cache(suppress_st_warning=True)
    def predict(self, txt):
        langs, probs = self.lid.predict(txt, k=1)
        sys.stdout.write(now+'Lid:\t{} {}\n'.format(langs[0][9:],probs[0]))
        return langs[0][9:], probs[0]
        

####################################################################################
### Build HTML with form ###########################################################
####################################################################################

st.set_page_config(page_title="GEC App", page_icon="🧠", layout="wide")
st.title("Grammatical Error Correction")

with st.form("my_form"):
    st.caption('Model')
    do_langid = st.checkbox('LangID', value=True)
    cols_fr = st.columns(3)
    do_encdec_fr = cols_fr[0].checkbox('EncDec_fr', value=True)
    do_srilm_fr = cols_fr[1].checkbox('SriLM_fr', value=False)
    do_gector_fr = cols_fr[2].checkbox('GECToR_fr', value=False)
    cols_en = st.columns(3)
    do_encdec_en = cols_en[0].checkbox('EncDec_en', value=False)
    do_srilm_en = cols_en[1].checkbox('SriLM_en', value=False)
    do_gector_en = cols_en[2].checkbox('GECToR_en', value=False)
    lsrc = st.text_area(label="Source", value='', max_chars=1000).rstrip().split('\n')
    submit = st.form_submit_button("Correct")

    
if submit:
    if do_langid:
        tic = time.time()
        if 'lid' not in st.session_state:        
            st.session_state.lid = Lid()
        language, prob = st.session_state.lid.predict(' '.join(lsrc))
        toc = time.time()
        st.write(now+'**LanguageID**: [{:.4f} sec]'.format(toc-tic))
        st.metric("({:.4f})".format(prob), language)

    sys.stdout.write(now+'Input:\t{}\n'.format(lsrc))

    if do_encdec_fr:
        tic = time.time()
        if 'encdec_fr' not in st.session_state:
            st.session_state.encdec_fr = EncDec('fr')
        if len(st.session_state.encdec_fr):
            lhyp = st.session_state.encdec_fr.correct(lsrc,now)
            toc = time.time()
            st.write(now+'**Enc-Dec_fr**: [{:.4f} sec]'.format(toc-tic))
            st.info(mark_diffs(lsrc, lhyp))
        else:
            st.write(now+'**Enc-Dec_fr** not available')

    if do_encdec_en:
        tic = time.time()
        if 'encdec_en' not in st.session_state:
            st.session_state.encdec_en = EncDec('en')
        if len(st.session_state.encdec_en):
            lhyp = st.session_state.encdec_en.correct(lsrc,now)
            toc = time.time()
            st.write(now+'**Enc-Dec_en**: [{:.4f} sec]'.format(toc-tic))
            st.info(mark_diffs(lsrc, lhyp))
        else:
            st.write(now+'**Enc-Dec_en** not available')

    if do_srilm_fr:
        tic = time.time()
        if 'srilm' not in st.session_state:
            st.session_state.srilm_fr = SriLM('fr')
        if len(st.session_state.srilm_fr):
            lhyp = st.session_state.srilm_fr.correct(lsrc,now)
            toc = time.time()
            st.write(now+'**SriLM_fr**: [{:.4f} sec]'.format(toc-tic))
            st.info(mark_diffs(lsrc, lhyp))
        else:
            st.write(now+'**SriLM_fr** not available')

    if do_srilm_en:
        tic = time.time()
        if 'srilm' not in st.session_state:
            st.session_state.srilm_en = SriLM('en')
        if len(st.session_state.srilm_en):
            lhyp = st.session_state.srilm_en.correct(lsrc,now)
            toc = time.time()
            st.write(now+'**SriLM_en**: [{:.4f} sec]'.format(toc-tic))
            st.info(mark_diffs(lsrc, lhyp))
        else:
            st.write(now+'**SriLM_en** not available')

    if do_gector_fr:
        tic = time.time()
        if 'gector_fr' not in st.session_state:
            st.session_state.gector_fr = GECToR('fr')
        if len(st.session_state.gector_fr):
            lhyp = st.session_state.gector_fr.correct(lsrc,now)
            toc = time.time()
            st.write(now+'**GECToR_fr**: [{:.4f} sec]'.format(toc-tic))
            st.info(mark_diffs(lsrc, lhyp))
        else:
            st.write(now+'**GECToR_fr** not available')

    if do_gector_en:
        tic = time.time()
        if 'gector_en' not in st.session_state:
            st.session_state.gector_en = GECToR('en')
        if len(st.session_state.gector_en):
            lhyp = st.session_state.gector_en.correct(lsrc,now)
            toc = time.time()
            st.write(now+'**GECToR_en**: [{:.4f} sec]'.format(toc-tic))
            st.info(mark_diffs(lsrc, lhyp))
        else:
            st.write(now+'**GECToR_en** not available')

        
