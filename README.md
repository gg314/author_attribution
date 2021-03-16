# A learning-Rust machine-learning project

| :warning: This project is a learning exercise. For example, external, numerical analysis packages have been avoided.  |
| :-- |

| :warning: This results of this project do not represent a serious attempt at literary analysis. Very little effort has been spent on verifying transcriptions or tracking down similar subject matters. Daniel Defoe's works are notoriously difficult to attribute, and traditional attributions made by humans should obviously be favored instead of this exercise.   |
| :-- |

## Goal
To someday say something provocative about some digitized works, especially about the attribution of *A General History of the Pyrates* (1724) to Daniel Defoe. Around 40 works (half from Defoe and half from others) dated 1700-1730 were obtained from Project Gutenberg. The text was analyzed in a variety of ways to build a statistical model for attributing unknown works to Defoe or to others.

## Repository contents
* Basic perceptron & testing with Fisher's famous [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris).
* Shallow feature-analysis on digitized texts (unique words, sentence length, hapax legomena, etc.)
* ... A full-featured SVM, eventually.
* ... Possibly a simple neural network approach, eventually.

### Useful references for SVMs and literary attributions
1. Joachims, Thorsten (1998). *Text Categorization with Support Vector Machines: Learning with Many Relevant Features*. ECML 1998: Machine Learning.
2. Joachims, Thorsten (1998). *Making Large-Scale SVM Learning Practical*. Advances in Kernal Methods - Support Vector Learning, MIT Press, Cambridge, USA.
3. Diederich, J., Kindermann, J., Leopold, E., and Paass, G. (2003). *Authorship Attribution with Support Vector Machines*. Applied Intelligence volume 19.

### Useful references for Defoe
1. Hargevik, Steig (1974). *The Disputed Assignment of Memoirs of an Enlglish Officer to Daniel Defoe*. Stockholm Studies in English, Almqvist & Wiksell.
2. Furbank, P.N. and Owens, W.R. (1998) *A Critical Bibliography of Daniel Defoe*. Pickering & Chatto Ltd. London.
3. Rothman, I.N. (2000). *Defoe De-Attributions Scrutinized under Hargevik Criteria: Applying Stylometrics to the Canon*. Bibliographical Society of America.