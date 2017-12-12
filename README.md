# CompareModels_TRECQA
In a QA system that needs to infer from unstructured corpus, one challenge is to choose the sentence that contains best answer information for the given question.

These files provides five baseline models, i.e. average pooling, RNN, CNN, RNNCNN, QA-LSTM/CNN+attention (Tan, 2015; state-of-art 2015), for the TrecQA task (wang et al. 2007).

Model Comparison
----------------
All models were trained on train-all using Keras 2.1.2.  
You can download the glove parameters at here http://nlp.stanford.edu/data/glove.6B.zip  
Batch normalization was used to improve the performance of the models over the results of the pasky's experiments.  
https://github.com/brmson/dataset-sts/tree/master/data/anssel/wang

If you see the other performance records on this dataset, visit here.
https://aclweb.org/aclwiki/Question_Answering_(State_of_the_art)

| Model                    | devMRR   | testMRR  | etc
|--------------------------|----------|----------|---------
| Avg.                     | 0.83315  | 0.807789 | pdim=0.5, Ddim=1
| CNN                      | 0.865507 | 0.859114 | pdim=0.5, p_layers=1, Ddim = 1
| RNN(LSTM)                | 0.842302 | 0.827154 | sdim=5~7, rnn=CuDNNLSTM, rnnbidi_mode=concatenate, Ddim = 2, proj=False
| RNN+CNN                  | 0.862692 | 0.803874 | Ddim=2, p_layers=2, pdim=0.5, rnn=CuDNNLSTM, rnnbidi_mode=concatenate sdim=1
| QA-LSTM/CNN+attention    | 0.832858 | 0.832281 | Ddim=[1, 1/2], p_layers=2, pdim=0.5, rnn=CuDNNLSTM, rnnbidi_mode=concatenate sdim=1, adim=0.5, state-of-art 2015

	
This year(2017)'s new results (TO DO list to implement)
----------------
| Model                    | testMRR  | etc
|--------------------------|----------|---------
| HyperQA                  |  0.865   | Tay et al. (2017)
| BiMPM                    |	0.875	  | Wang et al. (2017)
| Compare-Aggregate	       |	0.899   | Bian et al. (2017)
| IWAN	                   |	0.889   | Shen et al. (2017)


Reference
-----------------
- Ming Tan, Cicero dos Santos, Bing Xiang & Bowen Zhou. 2015. LSTM-Based Deep Learning Models for Nonfactoid Answer Selection. In eprint arXiv:1511.04108.
- Yi Tay, Luu Anh Tuan, Siu Cheung Hui. 2017 Enabling Efficient Question Answer Retrieval via Hyperbolic Neural Networks. In eprint arXiv: 1707.07847.
- Zhiguo Wang, Wael Hamza and Radu Florian. 2017. Bilateral Multi-Perspective Matching for Natural Language Sentences. In eprint arXiv:1702.03814.
- Weijie Bian, Si Li, Zhao Yang, Guang Chen, Zhiqing Lin. 2017. A Compare-Aggregate Model with Dynamic-Clip Attention for Answer Selection. In CIKM 2017.
- Gehui Shen, Yunlun Yang, Zhi-Hong Deng. 2017. Inter-Weighted Alignment Network for Sentence Pair Modeling. In EMNLP 2017.
