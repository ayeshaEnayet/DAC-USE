# Dialogue-Act-Classification-for-Dialogue-Analysis
This repository contains a trained model for the dialogue act classification of the utterances of dialogue. The classifier is constructed using the Universal Sentence Encoder available at Tensorflow-Hub (https://tfhub.dev/google/universal-sentence-encoder-large/2). The classifier is trained on SwDA dataset, and the file dialogue1.csv contains the utterances of the dataset. 
# Requirements:
  tensorflow: 1.15.0,
  tensorflow-hub: 0.5.0,
  keras: 2.3.1,
  h5py = 2.10.0,
  sklearn,
  pandas 
# Tagset:
https://web.stanford.edu/~jurafsky/ws97/manual.august1.html
# How it works:
Execute the file DAtagger.py, it will read the dialogue from test_file.csv and write the results to reult.File.
# DA Sequences:
This repository contains DA sequences from eight different Datasets in a file DomainShift.csv. The eight datasets are g: The GitHub dataset, ami: AMI Meeting dataset, d: Diplomacy dataset, h: Hate Speech dataset, t: Teams dataset,  s: SwDA dataset, m: SAMsum dataset, a: Military (Army) dataset. We randomly selected the subset of the datasets to generate the sequences. These sequences could be used to analyze domain shift, generalizability, and similarity of DA sequences across different datasets.
# GitHub Dataset:
This repository contains GitHub dataset. we manually labeled the dialogues as being high conflict or low conflict using the following criteria:
    The issue did not resolve successfully.
    The question(s) of the team member(s) remained unanswered.
    One or more team members did not understand the issue.
    Lack of understanding or disagreement between the team members.
    At least one team member did not agree with the suggested solution.
# Citation:
if you use this classifier, please cite it as:
@inproceedings{enayet2021analyzing,
  title={Analyzing Team Performance with Embeddings from Multiparty Dialogues},
  author={Enayet, Ayesha and Sukthankar, Gita},
  booktitle={2021 IEEE 15th International Conference on Semantic Computing (ICSC)},
  pages={33--39},
  year={2021},
  organization={IEEE}
}
Or
@article{enayet2020transfer,
  title={A Transfer Learning Approach for Dialogue Act Classification of GitHub Issue Comments},
  author={Enayet, Ayesha and Sukthankar, Gita},
  journal={arXiv preprint arXiv:2011.04867},
  year={2020}
}
# Relevant papers:
  Enayet, A., & Sukthankar, G. (2020). A Transfer Learning Approach for Dialogue Act Classification of GitHub Issue Comments. arXiv preprint arXiv:2011.04867.
  https://arxiv.org/pdf/2011.04867.pdf
  
 Enayet, A., & Sukthankar, G. (2021, January). Analyzing Team Performance with Embeddings from Multiparty Dialogues. In 2021 IEEE 15th International Conference on Semantic Computing (ICSC) (pp. 33-39). IEEE.
 https://ieeexplore.ieee.org/document/9364556
