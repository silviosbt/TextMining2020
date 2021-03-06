# An Intelligent System for the Categorization of Question Time Official Documents of the Italian Chamber of Deputies
<!-- *Alice Cavalieri, Pietro Ducange, Silvio Fabi, Federico Russo, Nicola Tonellotto* -->


In the following, we report the average values and standard deviations (over the 20 test sets of the cross validation) of precision, recall, and F1-measure per class and for each text classification model on the experimental dataset described in the manuscript.

## Models

- Bag of words (BOW):
    + Without feature selection:
        * [Support Vector Machines (SVM)](#bow_svm)
        * [Complement Naive Bayes (CNB)](#bow_cnb)
        * [Passive Aggressive Classifier (PAC)](#bow_pac)
        * [Multi-Layer Perceptron (MLP)](#bow_mlp)
    + With feature selection to 20,000 terms:
        * [Support Vector Machines (SVM)](#bow_svm_20000)
        * [Complement Naive Bayes (CNB)](#bow_cnb_20000)
        * [Passive Aggressive Classifier (PAC)](#bow_pac_20000)
        * [Multi-Layer Perceptron (MLP)](#bow_mlp_20000)
- Word embeddings:
    + Using Word2Vec:
        * [Convolutional Neural Network (CNN)](#word2vec_cnn)
        * [Long Short Term Memory (LSTM)](#word2vec_lstm)
    + Using FastText:
        * [Convolutiona Neural Network (CNN)](#fasttext_cnn)
        * [Long Short Term Memory (LSTM)](#fasttext_lstm)

### BOW_CNB_20000

| code | Macro topic                                       | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 58.4 (+/-6.1)  | 66.0 (+/-6.2)  | 61.9 (+/-5.9)  |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 67.4 (+/-17.4) | 27.7 (+/-10.7) | 38.4 (+/-11.6) |
| 3    | Health                                            | 83.3 (+/-4.8)  | 90.8 (+/-4.5)  | 86.8 (+/-4.0)  |
| 4    | Agriculture                                       | 77.9 (+/-6.6)  | 88.8 (+/-7.7)  | 82.9 (+/-6.3)  |
| 5    | Labour and Employment                             | 65.5 (+/-7.2)  | 65.2 (+/-6.8)  | 65.2 (+/-6.2)  |
| 6    | Education                                         | 72.9 (+/-6.3)  | 95.8 (+/-3.4)  | 82.6 (+/-4.1)  |
| 7    | Environment                                       | 68.6 (+/-4.0)  | 86.9 (+/-7.3)  | 76.5 (+/-4.0)  |
| 8    | Energy                                            | 73.7 (+/-11.7) | 78.1 (+/-16.5) | 75.5 (+/-13.3) |
| 9    | Immigration                                       | 75.0 (+/-6.8)  | 79.6 (+/-9.7)  | 77.0 (+/-6.9)  |
| 10   | Transportation                                    | 79.0 (+/-4.3)  | 92.2 (+/-3.2)  | 85.0 (+/-3.0)  |
| 12   | Low and Crime                                     | 71.5 (+/-3.9)  | 87.1 (+/-4.0)  | 78.5 (+/-2.1)  |
| 13   | Welfare                                           | 83.3 (+/-16.1) | 33.6 (+/-15.6) | 45.5 (+/-16.7) |
| 14   | C. Development and Housing Issue                 | 83.8 (+/-14.3) | 58.6 (+/-20.6) | 66.4 (+/-17.3) |
| 15   | Banking, Finance, and Domestic Commerce           | 68.7 (+/-7.4)  | 48.5 (+/-5.9)  | 56.7 (+/-5.5)  |
| 16   | Defence                                           | 76.0 (+/-15.4) | 52.0 (+/-10.9) | 61.0 (+/-10.5) |
| 17   | Space, Science, Technology, and Communications    | 73.2 (+/-11.4) | 64.5 (+/-13.1) | 67.9 (+/-10.4) |
| 18   | Foreign Trade                                     | 50.0 (+/-48.7) | 17.9 (+/-16.7) | 26.0 (+/-24.4) |
| 19   | International Affairs                             | 68.7 (+/-8.9)  | 62.7 (+/-8.1)  | 65.2 (+/-6.8)  |
| 20   | Government Operations                             | 67.9 (+/-7.8)  | 51.0 (+/-8.8)  | 58.0 (+/-7.7)  |
| 21   | Public Lands and Water Management                 | 71.4 (+/-14.2) | 53.1 (+/-13.2) | 59.8 (+/-11.6) |
| 23   | Cultural Policy Issues                            | 78.3 (+/-40.9) | 38.8 (+/-27.9) | 49.6 (+/-30.6) |

### BOW_SVM

| Code | Macro Topic                                       | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 57.2 (+/-6.5)  | 58.4 (+/-8.1)  | 57.7 (+/-6.9)  |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 65.2 (+/-18.1) | 39.7 (+/-9.5)  | 48.6 (+/-10.9) |
| 3    | Health                                            | 84.8 (+/-5.4)  | 86.5 (+/-6.4)  | 85.5 (+/-5.0)  |
| 4    | Agriculture                                       | 81.9 (+/-6.7)  | 82.4 (+/-7.3)  | 81.9 (+/-5.6)  |
| 5    | Labour and Employment                             | 64.6 (+/-6.6)  | 63.3 (+/-8.2)  | 63.6 (+/-5.9)  |
| 6    | Education                                         | 83.8 (+/-4.9)  | 87.5 (+/-6.1)  | 85.4 (+/-3.9)  |
| 7    | Environment                                       | 79.4 (+/-5.6)  | 77.3 (+/-6.6)  | 78.2 (+/-4.6)  |
| 8    | Energy                                            | 77.7 (+/-9.6)  | 76.3 (+/-10.6) | 76.6 (+/-8.0)  |
| 9    | Immigration                                       | 76.7 (+/-6.4)  | 74.5 (+/-9.3)  | 75.2 (+/-5.6)  |
| 10   | Transportation                                    | 84.2 (+/-4.9)  | 86.5 (+/-4.3)  | 85.3 (+/-3.6)  |
| 12   | Low and Crime                                     | 68.9 (+/-2.9)  | 81.8 (+/-6.2)  | 74.7 (+/-3.5)  |
| 13   | Welfare                                           | 62.3 (+/-11.7) | 52.0 (+/-13.5) | 55.8 (+/-11.1) |
| 14   | C. Development and Housing Issue                 | 76.8 (+/-18.0) | 54.8 (+/-19.2) | 61.9 (+/-15.8) |
| 15   | Banking, Finance, and Domestic Commerce           | 54.6 (+/-6.1)  | 54.9 (+/-7.1)  | 54.7 (+/-6.2)  |
| 16   | Defence                                           | 75.5 (+/-11.0) | 69.9 (+/-11.2) | 72.0 (+/-8.7)  |
| 17   | Space, Science, Technology, and Communications    | 72.0 (+/-10.4) | 59.4 (+/-15.7) | 63.8 (+/-11.2) |
| 18   | Foreign Trade                                     | 33.3 (+/-38.9) | 25.4 (+/-31.0) | 27.2 (+/-31.0) |
| 19   | International Affairs                             | 68.1 (+/-9.1)  | 63.3 (+/-7.2)  | 65.2 (+/-6.3)  |
| 20   | Government Operations                             | 61.3 (+/-6.1)  | 62.0 (+/-6.0)  | 61.5 (+/-4.9)  |
| 21   | Public Lands and Water Management                 | 71.3 (+/-13.9) | 54.9 (+/-16.4) | 61.1 (+/-14.0) |
| 23   | Cultural Policy Issues                            | 70.0 (+/-36.5) | 36.2 (+/-24.7) | 44.9 (+/-25.7) |

### BOW_CNB

| Code | Macro Topic                                       | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 57.8 (+/-6.9)  | 66.5 (+/-5.6)  | 61.7 (+/-6.0)  |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 70.5 (+/-15.1) | 30.0 (+/-10.0) | 41.4 (+/-11.1) |
| 3    | Health                                            | 83.6 (+/-4.8)  | 90.5 (+/-5.4)  | 86.8 (+/-4.1)  |
| 4    | Agriculture                                       | 79.3 (+/-7.1)  | 88.8 (+/-7.7)  | 83.6 (+/-6.0)  |
| 5    | Labour and Employment                             | 67.1 (+/-6.1)  | 66.1 (+/-7.1)  | 66.4 (+/-5.3)  |
| 6    | Education                                         | 74.8 (+/-6.3)  | 94.9 (+/-3.4)  | 83.5 (+/-4.2)  |
| 7    | Environment                                       | 75.2 (+/-5.6)  | 83.9 (+/-7.0)  | 79.1 (+/-4.5)  |
| 8    | Energy                                            | 77.3 (+/-11.6) | 82.0 (+/-14.5) | 79.2 (+/-11.8) |
| 9    | Immigration                                       | 75.1 (+/-6.4)  | 79.2 (+/-9.1)  | 76.8 (+/-6.3)  |
| 10   | Transportation                                    | 78.8 (+/-3.6)  | 92.0 (+/-3.5)  | 84.8 (+/-2.9)  |
| 12   | Low and Crime                                     | 71.1 (+/-3.6)  | 87.6 (+/-4.0)  | 78.5 (+/-3.1)  |
| 13   | Welfare                                           | 78.9 (+/-17.1) | 33.6 (+/-14.4) | 45.4 (+/-16.3) |
| 14   | C. Development and Housing Issue                 | 84.2 (+/-13.8) | 58.6 (+/-20.6) | 66.3 (+/-16.8) |
| 15   | Banking, Finance, and Domestic Commerce           | 65.5 (+/-5.5)  | 53.5 (+/-7.7)  | 58.6 (+/-5.5)  |
| 16   | Defence                                           | 78.4 (+/-13.0) | 53.1 (+/-9.0)  | 62.6 (+/-8.0)  |
| 17   | Space, Science, Technology, and Communications    | 78.7 (+/-14.2) | 60.6 (+/-14.1) | 67.3 (+/-11.3) |
| 18   | Foreign Trade                                     | 57.5 (+/-43.8) | 24.6 (+/-18.2) | 33.5 (+/-24.1) |
| 19   | International Affairs                             | 73.8 (+/-10.1) | 61.0 (+/-9.1)  | 66.3 (+/-7.7)  |
| 20   | Government Operations                             | 68.0 (+/-5.7)  | 54.2 (+/-9.0)  | 60.0 (+/-7.0)  |
| 21   | Public Lands and Water Management                 | 69.7 (+/-16.7) | 53.1 (+/-14.0) | 59.4 (+/-13.3) |
| 23   | Cultural Policy Issues                            | 70.8 (+/-43.9) | 34.2 (+/-28.2) | 43.8 (+/-30.9) |

### BOW_PAC

| Code | Macro Topic                                       | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 58.1 (+/-5.3)  | 60.4 (+/-6.5)  | 59.1 (+/-5.2)  |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 60.9 (+/-14.9) | 40.3 (+/-11.1) | 47.6 (+/-10.8) |
| 3    | Health                                            | 85.3 (+/-5.4)  | 88.5 (+/-5.4)  | 86.8 (+/-4.2)  |
| 4    | Agriculture                                       | 82.2 (+/-8.8)  | 83.7 (+/-8.3)  | 82.7 (+/-7.2)  |
| 5    | Labour and Employment                             | 64.1 (+/-7.5)  | 66.4 (+/-8.6)  | 64.9 (+/-6.6)  |
| 6    | Education                                         | 84.2 (+/-5.4)  | 87.9 (+/-5.5)  | 85.8 (+/-3.7)  |
| 7    | Environment                                       | 78.7 (+/-6.7)  | 78.7 (+/-5.7)  | 78.4 (+/-4.2)  |
| 8    | Energy                                            | 75.2 (+/-10.8) | 77.2 (+/-11.8) | 75.7 (+/-9.2)  |
| 9    | Immigration                                       | 77.1 (+/-6.0)  | 75.8 (+/-8.6)  | 76.2 (+/-5.7)  |
| 10   | Transportation                                    | 83.9 (+/-4.9)  | 88.3 (+/-3.8)  | 85.9 (+/-3.4)  |
| 12   | Low and Crime                                     | 74.0 (+/-3.2)  | 81.3 (+/-4.5)  | 77.4 (+/-3.2)  |
| 13   | Welfare                                           | 63.4 (+/-9.2)  | 52.6 (+/-14.2) | 56.3 (+/-10.4) |
| 14   | C. Development and Housing Issue                 | 74.5 (+/-16.3) | 57.5 (+/-20.8) | 62.8 (+/-17.0) |
| 15   | Banking, Finance, and Domestic Commerce           | 58.2 (+/-5.1)  | 54.8 (+/-8.3)  | 56.3 (+/-6.3)  |
| 16   | Defence                                           | 76.2 (+/-12.1) | 74.8 (+/-8.9)  | 74.7 (+/-7.4)  |
| 17   | Space, Science, Technology, and Communications    | 71.7 (+/-11.2) | 58.9 (+/-13.1) | 63.4 (+/-8.6)  |
| 18   | Foreign Trade                                     | 41.7 (+/-45.4) | 22.9 (+/-26.6) | 28.2 (+/-30.9) |
| 19   | International Affairs                             | 69.9 (+/-9.4)  | 67.2 (+/-8.3)  | 68.2 (+/-7.1)  |
| 20   | Government Operations                             | 60.6 (+/-5.9)  | 60.7 (+/-7.3)  | 60.5 (+/-5.7)  |
| 21   | Public Lands and Water Management                 | 69.9 (+/-12.7) | 57.7 (+/-13.1) | 62.2 (+/-10.3) |
| 23   | Cultural Policy Issues                            | 61.3 (+/-40.8) | 39.6 (+/-33.3) | 44.6 (+/-31.1) |

### BOW_MLP

| Code | Macro Topic                                       | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 56.0 (+/-7.1)  | 64.0 (+/-6.0)  | 59.6 (+/-5.8)  |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 66.1 (+/-15.1) | 43.0 (+/-11.5) | 51.2 (+/-10.9) |
| 3    | Health                                            | 85.9 (+/-5.6)  | 86.8 (+/-6.2)  | 86.1 (+/-4.1)  |
| 4    | Agriculture                                       | 83.0 (+/-8.9)  | 76.0 (+/-12.5) | 78.8 (+/-9.5)  |
| 5    | Labour and Employment                             | 69.0 (+/-6.8)  | 66.5 (+/-7.1)  | 67.5 (+/-5.4)  |
| 6    | Education                                         | 85.5 (+/-4.6)  | 85.5 (+/-6.6)  | 85.3 (+/-4.4)  |
| 7    | Environment                                       | 79.8 (+/-6.2)  | 77.7 (+/-6.2)  | 78.5 (+/-4.6)  |
| 8    | Energy                                            | 80.2 (+/-13.1) | 74.1 (+/-12.4) | 76.2 (+/-9.9)  |
| 9    | Immigration                                       | 80.4 (+/-7.1)  | 76.0 (+/-8.7)  | 77.7 (+/-5.3)  |
| 10   | Transportation                                    | 86.3 (+/-3.9)  | 86.5 (+/-4.0)  | 86.4 (+/-3.2)  |
| 12   | Low and Crime                                     | 72.6 (+/-3.6)  | 85.8 (+/-4.0)  | 78.6 (+/-2.9)  |
| 13   | Welfare                                           | 63.5 (+/-13.5) | 57.0 (+/-15.1) | 59.0 (+/-11.8) |
| 14   | C. Development and Housing Issue                 | 79.5 (+/-19.0) | 50.4 (+/-18.6) | 59.0 (+/-15.4) |
| 15   | Banking, Finance, and Domestic Commerce           | 56.6 (+/-7.0)  | 59.2 (+/-6.6)  | 57.5 (+/-4.9)  |
| 16   | Defence                                           | 76.6 (+/-8.1)  | 59.0 (+/-10.4) | 66.1 (+/-7.9)  |
| 17   | Space, Science, Technology, and Communications    | 79.5 (+/-17.4) | 50.3 (+/-14.7) | 59.9 (+/-13.4) |
| 18   | Foreign Trade                                     | 32.5 (+/-44.1) | 19.2 (+/-26.1) | 22.7 (+/-29.8) |
| 19   | International Affairs                             | 71.0 (+/-9.2)  | 66.1 (+/-7.1)  | 68.0 (+/-5.9)  |
| 20   | Government Operations                             | 60.2 (+/-5.9)  | 64.2 (+/-7.4)  | 62.0 (+/-5.4)  |
| 21   | Public Lands and Water Management                 | 63.3 (+/-12.7) | 54.0 (+/-16.0) | 57.6 (+/-13.9) |
| 23   | Cultural Policy Issues                            | 43.3 (+/-47.0) | 21.2 (+/-25.1) | 27.3 (+/-30.5) |


### BOW_SVM_20000

| Code | Macro Topic                                       | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 48.2 (+/-4.0)  | 52.2 (+/-6.6)  | 49.9 (+/-4.3)  |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 53.4 (+/-14.6) | 36.0 (+/-10.9) | 42.3 (+/-10.5) |
| 3    | Health                                            | 76.3 (+/-5.9)  | 78.2 (+/-6.6)  | 77.0 (+/-4.5)  |
| 4    | Agriculture                                       | 72.6 (+/-8.9)  | 68.1 (+/-10.5) | 70.0 (+/-8.6)  |
| 5    | Labour and Employment                             | 58.2 (+/-5.2)  | 57.4 (+/-8.1)  | 57.5 (+/-5.4)  |
| 6    | Education                                         | 83.1 (+/-7.3)  | 78.3 (+/-7.1)  | 80.4 (+/-5.4)  |
| 7    | Environment                                       | 79.8 (+/-7.2)  | 71.8 (+/-7.4)  | 75.3 (+/-5.8)  |
| 8    | Energy                                            | 70.6 (+/-13.4) | 66.0 (+/-10.6) | 67.7 (+/-10.2) |
| 9    | Immigration                                       | 72.5 (+/-6.8)  | 65.8 (+/-11.6) | 68.5 (+/-8.1)  |
| 10   | Transportation                                    | 78.4 (+/-5.5)  | 79.2 (+/-4.4)  | 78.6 (+/-3.7)  |
| 12   | Low and Crime                                     | 61.9 (+/-4.7)  | 74.3 (+/-6.5)  | 67.4 (+/-4.7)  |
| 13   | Welfare                                           | 49.0 (+/-14.1) | 45.2 (+/-18.3) | 46.0 (+/-15.2) |
| 14   | C. Development and Housing Issue                 | 58.3 (+/-16.4) | 45.6 (+/-18.5) | 49.2 (+/-16.8) |
| 15   | Banking, Finance, and Domestic Commerce           | 40.2 (+/-7.2)  | 45.2 (+/-6.0)  | 42.4 (+/-6.0)  |
| 16   | Defence                                           | 68.8 (+/-14.3) | 60.2 (+/-17.5) | 63.4 (+/-13.8) |
| 17   | Space, Science, Technology, and Communications    | 64.9 (+/-14.7) | 46.5 (+/-12.8) | 53.2 (+/-12.4) |
| 18   | Foreign Trade                                     | 19.5 (+/-33.6) | 15.8 (+/-28.3) | 15.7 (+/-25.3) |
| 19   | International Affairs                             | 61.9 (+/-8.7)  | 55.8 (+/-9.3)  | 58.1 (+/-7.2)  |
| 20   | Government Operations                             | 54.3 (+/-6.5)  | 53.9 (+/-6.2)  | 53.9 (+/-5.5)  |
| 21   | Public Lands and Water Management                 | 55.2 (+/-15.3) | 43.7 (+/-16.9) | 47.9 (+/-16.0) |
| 23   | Cultural Policy Issues                            | 35.3 (+/-39.0) | 18.7 (+/-19.8) | 23.0 (+/-22.9) |

### BOW_PAC_20000

| Code | Macro Topic                                       | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 48.9 (+/-6.4)  | 54.8 (+/-9.1)  | 51.3 (+/-6.2)  |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 56.3 (+/-11.6) | 37.7 (+/-7.9)  | 44.4 (+/-7.4)  |
| 3    | Health                                            | 78.9 (+/-5.5)  | 82.4 (+/-6.2)  | 80.5 (+/-4.8)  |
| 4    | Agriculture                                       | 78.0 (+/-9.6)  | 74.1 (+/-10.9) | 75.6 (+/-8.5)  |
| 5    | Labour and Employment                             | 59.0 (+/-5.5)  | 60.8 (+/-9.3)  | 59.6 (+/-5.9)  |
| 6    | Education                                         | 81.9 (+/-6.4)  | 81.2 (+/-8.2)  | 81.2 (+/-5.1)  |
| 7    | Environment                                       | 79.9 (+/-5.9)  | 72.4 (+/-7.1)  | 75.8 (+/-5.8)  |
| 8    | Energy                                            | 79.9 (+/-12.7) | 65.5 (+/-15.4) | 70.9 (+/-12.4) |
| 9    | Immigration                                       | 76.0 (+/-6.0)  | 70.6 (+/-10.0) | 72.9 (+/-6.9)  |
| 10   | Transportation                                    | 78.6 (+/-5.0)  | 82.7 (+/-4.9)  | 80.5 (+/-4.0)  |
| 12   | Low and Crime                                     | 68.4 (+/-4.9)  | 78.3 (+/-4.1)  | 72.9 (+/-4.0)  |
| 13   | Welfare                                           | 53.3 (+/-12.9) | 44.8 (+/-17.6) | 47.6 (+/-14.7) |
| 14   | C. Development and Housing Issue                 | 66.9 (+/-29.7) | 39.6 (+/-20.4) | 47.9 (+/-22.0) |
| 15   | Banking, Finance, and Domestic Commerce           | 49.5 (+/-6.7)  | 51.0 (+/-5.9)  | 50.1 (+/-5.6)  |
| 16   | Defence                                           | 71.9 (+/-10.6) | 65.9 (+/-14.3) | 68.0 (+/-10.7) |
| 17   | Space, Science, Technology, and Communications    | 72.9 (+/-17.2) | 45.4 (+/-14.9) | 54.8 (+/-14.4) |
| 18   | Foreign Trade                                     | 23.3 (+/-42.0) | 9.6 (+/-18.6)  | 12.8 (+/-23.2) |
| 19   | International Affairs                             | 64.5 (+/-7.9)  | 61.3 (+/-9.7)  | 62.4 (+/-7.0)  |
| 20   | Government Operations                             | 52.1 (+/-6.4)  | 56.8 (+/-6.5)  | 54.2 (+/-5.7)  |
| 21   | Public Lands and Water Management                 | 63.6 (+/-18.6) | 46.4 (+/-15.1) | 52.6 (+/-15.1) |
| 23   | Cultural Policy Issues                            | 35.0 (+/-46.2) | 11.2 (+/-14.4) | 16.7 (+/-21.2) |

### BOW_MLP_20000

| Code | Macro Topic                                       | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 56.3 (+/-7.9)  | 62.4 (+/-6.2)  | 58.9 (+/-5.9)  |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 62.0 (+/-14.3) | 38.3 (+/-10.8) | 46.1 (+/-9.6)  |
| 3    | Health                                            | 84.3 (+/-5.4)  | 85.2 (+/-6.7)  | 84.6 (+/-4.5)  |
| 4    | Agriculture                                       | 82.0 (+/-8.3)  | 79.6 (+/-10.0) | 80.6 (+/-8.2)  |
| 5    | Labour and Employment                             | 67.2 (+/-6.0)  | 64.7 (+/-10.7) | 65.4 (+/-6.9)  |
| 6    | Education                                         | 82.2 (+/-5.2)  | 86.3 (+/-6.1)  | 84.0 (+/-4.1)  |
| 7    | Environment                                       | 80.2 (+/-7.2)  | 74.2 (+/-8.3)  | 76.8 (+/-6.2)  |
| 8    | Energy                                            | 81.1 (+/-14.0) | 66.8 (+/-16.1) | 72.3 (+/-13.1) |
| 9    | Immigration                                       | 77.3 (+/-7.8)  | 73.6 (+/-10.4) | 74.9 (+/-6.9)  |
| 10   | Transportation                                    | 84.5 (+/-4.1)  | 86.9 (+/-4.7)  | 85.6 (+/-3.2)  |
| 12   | Low and Crime                                     | 73.1 (+/-3.6)  | 84.1 (+/-5.1)  | 78.0 (+/-2.2)  |
| 13   | Welfare                                           | 61.2 (+/-12.0) | 50.0 (+/-12.8) | 53.6 (+/-9.0)  |
| 14   | C. Development and Housing Issue                 | 69.7 (+/-25.6) | 44.5 (+/-22.4) | 52.2 (+/-22.4) |
| 15   | Banking, Finance, and Domestic Commerce           | 57.0 (+/-6.3)  | 60.3 (+/-9.1)  | 58.1 (+/-5.5)  |
| 16   | Defence                                           | 79.0 (+/-10.6) | 64.4 (+/-8.8)  | 70.5 (+/-7.7)  |
| 17   | Space, Science, Technology, and Communications    | 80.6 (+/-13.0) | 47.3 (+/-13.0) | 58.5 (+/-11.6) |
| 18   | Foreign Trade                                     | 22.5 (+/-41.3) | 7.9 (+/-14.2)  | 11.5 (+/-20.6) |
| 19   | International Affairs                             | 68.2 (+/-7.2)  | 65.3 (+/-8.4)  | 66.4 (+/-6.2)  |
| 20   | Government Operations                             | 55.5 (+/-6.9)  | 62.9 (+/-7.2)  | 58.7 (+/-5.7)  |
| 21   | Public Lands and Water Management                 | 68.8 (+/-14.6) | 52.8 (+/-12.3) | 58.8 (+/-11.1) |
| 23   | Cultural Policy Issues                            | 22.5 (+/-41.3) | 10.0 (+/-19.6) | 13.5 (+/-25.6) |

### Word2Vec_CNN

| Code | Macro Topics                                      | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 47.9 (+/-0.09) | 51.8 (+/-0.16) | 47.2 (+/-0.09) |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 44.9 (+/-0.11) | 37.5 (+/-0.13) | 38.2 (+/-0.08) |
| 3    | Health                                            | 81.1 (+/-0.06) | 84.0 (+/-0.07) | 82.9 (+/-0.04) |
| 4    | Agriculture                                       | 80.3 (+/-0.11) | 74.8 (+/-0.11) | 76.6 (+/-0.07) |
| 5    | Labour and Employment                             | 61.1 (+/-0.12) | 61.9 (+/-0.17) | 59.6 (+/-0.12) |
| 6    | Education                                         | 82.3 (+/-0.09) | 82.8 (+/-0.1)  | 82.7 (+/-0.07) |
| 7    | Environment                                       | 75.0 (+/-0.14) | 71.9 (+/-0.13) | 72.7 (+/-0.08) |
| 8    | Energy                                            | 70.4 (+/-0.14) | 66.5 (+/-0.17) | 66.5 (+/-0.13) |
| 9    | Immigration                                       | 74.6 (+/-0.12) | 66.8 (+/-0.12) | 69.6 (+/-0.08) |
| 10   | Transportation                                    | 81.2 (+/-0.06) | 81.2 (+/-0.06) | 81.0 (+/-0.03) |
| 12   | Low and Crime                                     | 74.7 (+/-0.08) | 77.2 (+/-0.09) | 75.7 (+/-0.04) |
| 13   | Welfare                                           | 48.5 (+/-0.21) | 37.8 (+/-0.19) | 37.7 (+/-0.14) |
| 14   | C. Development and Housing Issue                  | 45.6 (+/-0.23) | 28.1 (+/-0.19) | 32.6 (+/-0.17) |
| 15   | Banking, Finance, and Domestic Commerce           | 51.6 (+/-0.16) | 48.2 (+/-0.17) | 46.4 (+/-0.09) |
| 16   | Defence                                           | 67.6 (+/-0.12) | 54.5 (+/-0.16) | 57.6 (+/-0.1)  |
| 17   | Space, Science, Technology, and Communications    | 63.4 (+/-0.21) | 45.5 (+/-0.17) | 50.4 (+/-0.14) |
| 18   | Foreign Trade                                     | 31.5 (+/-0.41) | 17.0 (+/-0.21) | 21.9 (+/-0.26) |
| 19   | International Affairs                             | 61.1 (+/-0.09) | 57.5 (+/-0.18) | 56.9 (+/-0.1)  |
| 20   | Government Operations                             | 56.7 (+/-0.08) | 53.5 (+/-0.1)  | 53.7 (+/-0.06) |
| 21   | Public Lands and Water Management                 | 50.5 (+/-0.18) | 44.3 (+/-0.14) | 44.5 (+/-0.12) |
| 23   | Cultural Policy Issues                            | 12.1 (+/-0.25) | 09.5 (+/-0.14) | 10.9 (+/-0.16) |

### FastText_CNN

| Code | Macro Topics                                      | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 50.9 (+/-0.1)  | 51.0 (+/-0.13) | 48.4 (+/-0.06) |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 50.1 (+/-0.2)  | 39.6 (+/-0.11) | 41.1 (+/-0.09) |
| 3    | Health                                            | 83.0 (+/-0.06) | 83.3 (+/-0.08) | 82.5 (+/-0.05) |
| 4    | Agriculture                                       | 78.6 (+/-0.09) | 69.4 (+/-0.13) | 73.1 (+/-0.09) |
| 5    | Labour and Employment                             | 65.4 (+/-0.13) | 60.7 (+/-0.09) | 62.9 (+/-0.08) |
| 6    | Education                                         | 80.3 (+/-0.09) | 81.3 (+/-0.1)  | 80.1 (+/-0.06) |
| 7    | Environment                                       | 72.3 (+/-0.11) | 71.6 (+/-0.13) | 70.3 (+/-0.07) |
| 8    | Energy                                            | 67.4 (+/-0.13) | 64.3 (+/-0.14) | 64.6 (+/-0.1)  |
| 9    | Immigration                                       | 74.8 (+/-0.09) | 64.6 (+/-0.15) | 67.7 (+/-0.09) |
| 10   | Transportation                                    | 82.2 (+/-0.06) | 83.9 (+/-0.08) | 82.0 (+/-0.04) |
| 12   | Low and Crime                                     | 72.8 (+/-0.08) | 79.8 (+/-0.09) | 75.2 (+/-0.05) |
| 13   | Welfare                                           | 39.8 (+/-0.23) | 38.4 (+/-0.22) | 36.1 (+/-0.18) |
| 14   | C. Development and Housing Issue                  | 36.4 (+/-0.34) | 20.2 (+/-0.18) | 24.4 (+/-0.2)  |
| 15   | Banking, Finance, and Domestic Commerce           | 47.4 (+/-0.09) | 45.2 (+/-0.15) | 44.8 (+/-0.08) |
| 16   | Defence                                           | 66.5 (+/-0.16) | 52.1 (+/-0.19) | 56.3 (+/-0.16) |
| 17   | Space, Science, Technology, and Communications    | 64.3 (+/-0.26) | 44.3 (+/-0.24) | 46.8 (+/-0.17) |
| 18   | Foreign Trade                                     | 29.0 (+/-0.38) | 20.7 (+/-0.25) | 21.3 (+/-0.25) |
| 19   | International Affairs                             | 62.7 (+/-0.1)  | 57.0 (+/-0.12) | 58.0 (+/-0.07) |
| 20   | Government Operations                             | 54.4 (+/-0.1)  | 58.9 (+/-0.09) | 55.1 (+/-0.04) |
| 21   | Public Lands and Water Management                 | 49.9 (+/-0.11) | 42.0 (+/-0.17) | 42.8 (+/-0.12) |
| 23   | Cultural Policy Issues                            | 20.8 (+/-0.36) | 10.6 (+/-0.17) | 12.4 (+/-0.21) |

### Word2Vec_LSTM

| Code | Macro Topics                                      | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 47.6 (+/-0.11) | 52.9 (+/-0.08) | 49.3 (+/-0.07) |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 43.2 (+/-0.36) | 21.1 (+/-0.16) | 28.8 (+/-0.22) |
| 3    | Health                                            | 81.0 (+/-0.06) | 88.5 (+/-0.04) | 84.9 (+/-0.03) |
| 4    | Agriculture                                       | 79.1 (+/-0.1)  | 90.6 (+/-0.07) | 84.0 (+/-0.07) |
| 5    | Labour and Employment                             | 64.7 (+/-0.06) | 65.6 (+/-0.14) | 63.0 (+/-0.07) |
| 6    | Education                                         | 78.6 (+/-0.11) | 92.4 (+/-0.1)  | 84.1 (+/-0.06) |
| 7    | Environment                                       | 74.7 (+/-0.06) | 77.8 (+/-0.1)  | 75.1 (+/-0.08) |
| 8    | Energy                                            | 70.4 (+/-0.1)  | 76.6 (+/-0.11) | 72.7 (+/-0.07) |
| 9    | Immigration                                       | 71.7 (+/-0.04) | 70.5 (+/-0.1)  | 70.6 (+/-0.07) |
| 10   | Transportation                                    | 82.5 (+/-0.06) | 83.8 (+/-0.04) | 83.0 (+/-0.04) |
| 12   | Low and Crime                                     | 76.4 (+/-0.05) | 82.9 (+/-0.07) | 78.2 (+/-0.04) |
| 13   | Welfare                                           | 29.2 (+/-0.22) | 17.3 (+/-0.13) | 21.6 (+/-0.15) |
| 14   | C. Development and Housing Issue                  | 23.1 (+/-0.35) | 12.2 (+/-0.22) | 16.4 (+/-0.27) |
| 15   | Banking, Finance, and Domestic Commerce           | 49.5 (+/-0.12) | 48.3 (+/-0.1)  | 48.1 (+/-0.1)  |
| 16   | Defence                                           | 62.5 (+/-0.09) | 54.9 (+/-0.21) | 54.6 (+/-0.12) |
| 17   | Space, Science, Technology, and Communications    | 68.2 (+/-0.21) | 37.3 (+/-0.21) | 44.0 (+/-0.17) |
| 18   | Foreign Trade                                     | 00.9 (+/-0.0)  | 00.3 (+/-0.0)  | 00.4 (+/-0.0)  |
| 19   | International Affairs                             | 54.1 (+/-0.04) | 56.5 (+/-0.12) | 55.9 (+/-0.07) |
| 20   | Government Operations                             | 55.0 (+/-0.07) | 56.4 (+/-0.07) | 55.9 (+/-0.03) |
| 21   | Public Lands and Water Management                 | 60.7 (+/-0.18) | 53.7 (+/-0.2)  | 54.9 (+/-0.16) |
| 23   | Cultural Policy Issues                            | 00.4 (+/-0.0)  | 00.5 (+/-0.0)  | 00.9 (+/-0.0)  |

### FastText_LSTM

| Code | Macro Topics                                      | Precision            | Recall               | F1-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 51.6 (+/-0.07) | 53.1 (+/-0.1)  | 51.0 (+/-0.06) |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 58.8 (+/-0.28) | 19.7 (+/-0.1)  | 26.1 (+/-0.12) |
| 3    | Health                                            | 81.2 (+/-0.04) | 89.1 (+/-0.06) | 85.2 (+/-0.04) |
| 4    | Agriculture                                       | 73.3 (+/-0.08) | 86.8 (+/-0.07) | 79.2 (+/-0.04) |
| 5    | Labour and Employment                             | 63.4 (+/-0.08) | 74.3 (+/-0.08) | 68.6 (+/-0.07) |
| 6    | Education                                         | 78.0 (+/-0.08) | 91.3 (+/-0.07) | 84.6 (+/-0.06) |
| 7    | Environment                                       | 70.0 (+/-0.06) | 71.3 (+/-0.22) | 68.6 (+/-0.12) |
| 8    | Energy                                            | 65.0 (+/-0.13) | 74.8 (+/-0.05) | 69.0 (+/-0.09) |
| 9    | Immigration                                       | 70.3 (+/-0.12) | 68.3 (+/-0.13) | 69.1 (+/-0.11) |
| 10   | Transportation                                    | 82.6 (+/-0.02) | 84.4 (+/-0.02) | 83.6 (+/-0.02) |
| 12   | Low and Crime                                     | 79.4 (+/-0.05) | 81.0 (+/-0.07) | 80.4 (+/-0.03) |
| 13   | Welfare                                           | 40.4 (+/-0.18) | 30.7 (+/-0.14) | 34.4 (+/-0.15) |
| 14   | C. Development and Housing Issue                  | 47.1 (+/-0.51) | 11.9 (+/-0.11) | 17.4 (+/-0.17) |
| 15   | Banking, Finance, and Domestic Commerce           | 52.8 (+/-0.15) | 44.4 (+/-0.14) | 46.1 (+/-0.11) |
| 16   | Defence                                           | 65.8 (+/-0.2)  | 54.6 (+/-0.19) | 56.6 (+/-0.14) |
| 17   | Space, Science, Technology, and Communications    | 74.4 (+/-0.17) | 40.0 (+/-0.12) | 51.4 (+/-0.08) |
| 18   | Foreign Trade                                     | 00.9 (+/-0.0)  | 00.0 (+/-0.0)  | 00.0 (+/-0.0)  |
| 19   | International Affairs                             | 58.9 (+/-0.06) | 61.3 (+/-0.11) | 59.4 (+/-0.08) |
| 20   | Government Operations                             | 54.2 (+/-0.07) | 56.6 (+/-0.07) | 55.2 (+/-0.05) |
| 21   | Public Lands and Water Management                 | 55.0 (+/-0.11) | 60.8 (+/-0.19) | 56.0 (+/-0.12) |
| 23   | Cultural Policy Issues                            | 00.5 (+/-0.0)  | 00.5 (+/-0.0)  | 00.6 (+/-0.0)  |




