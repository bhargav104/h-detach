Microsoft COCO Caption Evaluation
===================

Evaluation codes for MS COCO caption generation.
 
After generating features using a Resnet-152. These are the following commands to run the captioning experiment codes. Replace "folder" with the name of the corresponding folder where that file is present.

Commands 
- Show and Tell - python train.py --id st --caption_model show_tell --input_json folder/cocotalk.json --input_fc_dir folder/cocotalk_fc --input_att_dir folder/cocotalk_att --input_label_h5 /folder/cocotalk_label.h5 --batch_size 10  --checkpoint_path logs/coco_152_st_baseline --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30 --language_eval 1 --learning_rate 0.0001 --save_dir=st-baseline --p-detach=0.4

- Show Attend Tell - python train.py --id sat --caption_model show_attend_tell --input_json /folder/cocotalk.json --input_fc_dir /folder/cocotalk_fc --input_att_dir /folder/cocotalk_att --input_label_h5 folder/cocotalk_label.h5 --batch_size 10  --checkpoint_path logs/coco_152_sat_baseline --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30 --language_eval 1 --learning_rate 0.0001 --save_dir=sat-baseline --p-detach=0.4


## Requirements ##
- java 1.8.0
- python 2.7

## Files ##
- cocoEvalCapDemo.py (demo script)

./annotation
- captions_val2014.json (MS COCO 2014 caption validation set)
- Visit MS COCO [download](http://mscoco.org/dataset/#download) page for more details.

./results
- captions_val2014_fakecap_results.json (an example of fake results for running demo)
- Visit MS COCO [format](http://mscoco.org/dataset/#format) page for more details.

./pycocoevalcap: The folder where all evaluation codes are stored.
- evals.py: The file includes COCOEavlCap class that can be used to evaluate results on COCO.
- tokenizer: Python wrapper of Stanford CoreNLP PTBTokenizer
- bleu: Bleu evalutation codes
- meteor: Meteor evaluation codes
- rouge: Rouge-L evaluation codes
- cider: CIDEr evaluation codes
- spice: SPICE evaluation codes

## Setup ##

- You will first need to download the [Stanford CoreNLP 3.6.0](http://stanfordnlp.github.io/CoreNLP/index.html) code and models for use by SPICE. To do this, run:
    ./get_stanford_models.sh
- Note: SPICE will try to create a cache of parsed sentences in ./pycocoevalcap/spice/cache/. This dramatically speeds up repeated evaluations. The cache directory can be moved by setting 'CACHE_DIR' in ./pycocoevalcap/spice. In the same file, caching can be turned off by removing the '-cache' argument to 'spice_cmd'. 

## References ##

- [Microsoft COCO Captions: Data Collection and Evaluation Server](http://arxiv.org/abs/1504.00325)
- PTBTokenizer: We use the [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml) which is included in [Stanford CoreNLP 3.4.1](http://nlp.stanford.edu/software/corenlp.shtml).
- BLEU: [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf)
- Meteor: [Project page](http://www.cs.cmu.edu/~alavie/METEOR/) with related publications. We use the latest version (1.5) of the [Code](https://github.com/mjdenkowski/meteor). Changes have been made to the source code to properly aggreate the statistics for the entire corpus.
- Rouge-L: [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf)
- CIDEr: [CIDEr: Consensus-based Image Description Evaluation](http://arxiv.org/pdf/1411.5726.pdf)
- SPICE: [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/abs/1607.08822)

## Developers ##
- Xinlei Chen (CMU)
- Hao Fang (University of Washington)
- Tsung-Yi Lin (Cornell)
- Ramakrishna Vedantam (Virgina Tech)

## Acknowledgement ##
- David Chiang (University of Norte Dame)
- Michael Denkowski (CMU)
- Alexander Rush (Harvard University)
