# TextMining2020
New repository for text mining analysis

## BOW_SVM: performance scores on experimental dataset

| code | Macro topic                                       | Precision            | Recall               | F1\-measure          |
|------|---------------------------------------------------|----------------------|----------------------|----------------------|
| 1    | Domestic Microeconomic Issues                     | 58\.4 \(\+/\-6\.1\)  | 66\.0 \(\+/\-6\.2\)  | 61\.9 \(\+/\-5\.9\)  |
| 2    | Civil Right, Minority Issues, and Civil Liberties | 67\.4 \(\+/\-17\.4\) | 27\.7 \(\+/\-10\.7\) | 38\.4 \(\+/\-11\.6\) |
| 3    | Health                                            | 83\.3 \(\+/\-4\.8\)  | 90\.8 \(\+/\-4\.5\)  | 86\.8 \(\+/\-4\.0\)  |
| 4    | Agriculture                                       | 77\.9 \(\+/\-6\.6\)  | 88\.8 \(\+/\-7\.7\)  | 82\.9 \(\+/\-6\.3\)  |
| 5    | Labour and Employment                             | 65\.5 \(\+/\-7\.2\)  | 65\.2 \(\+/\-6\.8\)  | 65\.2 \(\+/\-6\.2\)  |
| 6    | Education                                         | 72\.9 \(\+/\-6\.3\)  | 95\.8 \(\+/\-3\.4\)  | 82\.6 \(\+/\-4\.1\)  |
| 7    | Environment                                       | 68\.6 \(\+/\-4\.0\)  | 86\.9 \(\+/\-7\.3\)  | 76\.5 \(\+/\-4\.0\)  |
| 8    | Energy                                            | 73\.7 \(\+/\-11\.7\) | 78\.1 \(\+/\-16\.5\) | 75\.5 \(\+/\-13\.3\) |
| 9    | Immigration                                       | 75\.0 \(\+/\-6\.8\)  | 79\.6 \(\+/\-9\.7\)  | 77\.0 \(\+/\-6\.9\)  |
| 10   | Transportation                                    | 79\.0 \(\+/\-4\.3\)  | 92\.2 \(\+/\-3\.2\)  | 85\.0 \(\+/\-3\.0\)  |
| 12   | Low and Crime                                     | 71\.5 \(\+/\-3\.9\)  | 87\.1 \(\+/\-4\.0\)  | 78\.5 \(\+/\-2\.1\)  |
| 13   | Welfare                                           | 83\.3 \(\+/\-16\.1\) | 33\.6 \(\+/\-15\.6\) | 45\.5 \(\+/\-16\.7\) |
| 14   | C\. Development and Housing Issue                 | 83\.8 \(\+/\-14\.3\) | 58\.6 \(\+/\-20\.6\) | 66\.4 \(\+/\-17\.3\) |
| 15   | Banking, Finance, and Domestic Commerce           | 68\.7 \(\+/\-7\.4\)  | 48\.5 \(\+/\-5\.9\)  | 56\.7 \(\+/\-5\.5\)  |
| 16   | Defence                                           | 76\.0 \(\+/\-15\.4\) | 52\.0 \(\+/\-10\.9\) | 61\.0 \(\+/\-10\.5\) |
| 17   | Space, Science, Technology, and Communications    | 73\.2 \(\+/\-11\.4\) | 64\.5 \(\+/\-13\.1\) | 67\.9 \(\+/\-10\.4\) |
| 18   | Foreign Trade                                     | 50\.0 \(\+/\-48\.7\) | 17\.9 \(\+/\-16\.7\) | 26\.0 \(\+/\-24\.4\) |
| 19   | International Affairs                             | 68\.7 \(\+/\-8\.9\)  | 62\.7 \(\+/\-8\.1\)  | 65\.2 \(\+/\-6\.8\)  |
| 20   | Government Operations                             | 67\.9 \(\+/\-7\.8\)  | 51\.0 \(\+/\-8\.8\)  | 58\.0 \(\+/\-7\.7\)  |
| 21   | Public Lands and Water Management                 | 71\.4 \(\+/\-14\.2\) | 53\.1 \(\+/\-13\.2\) | 59\.8 \(\+/\-11\.6\) |
| 23   | Cultural Policy Issues                            | 78\.3 \(\+/\-40\.9\) | 38\.8 \(\+/\-27\.9\) | 49\.6 \(\+/\-30\.6\) |


## BOW_CNB_20000: performance scores on test dataset

| Code           | Macro Topics                                      | Precision           | Recall           | F1-measure   | Supp            |
|----------------|---------------------------------------------------|---------------------|------------------|--------------|-----------------|
| 1              | Domestic Microeconomic Issues                     | 0\.58               | 0\.65            | 0\.61        | 17              |
| 2              | Civil Right, Minority Issues, and Civil Liberties | 1\.0                | 0\.5             | 0\.67        | 4               |
| 3              | Health                                            | 0\.79               | 1\.0             | 0\.88        | 15              |
| 4              | Agriculture                                       | 0\.71               | 0\.83            | 0\.77        | 6               |
| 5              | Labour and Employment                             | 0\.57               | 1\.0             | 0\.73        | 4               |
| 6              | Education                                         | 1\.0                | 1\.0             | 1\.0         | 13              |
| 7              | Environment                                       | 0\.62               | 0\.71            | 0\.67        | 14              |
| 8              | Energy                                            | 0\.67               | 1\.0             | 0\.8         | 2               |
| 9              | Immigration                                       | 0\.73               | 1\.0             | 0\.84        | 8               |
| 10             | Transportation                                    | 0\.77               | 0\.91            | 0\.83        | 33              |
| 12             | Low and Crime                                     | 0\.83               | 0\.96            | 0\.89        | 25              |
| 13             | Welfare                                           | 1\.0                | 0\.4             | 0\.57        | 5               |
| 14             | C\. Development and Housing Issue                 | 0\.0                | 0\.0             | 0\.0         | 4               |
| 15             | Banking, Finance, and Domestic Commerce           | 0\.47               | 0\.47            | 0\.47        | 15              |
| 16             | Defence                                           | 1\.0                | 0\.5             | 0\.67        | 2               |
| 17             | Space, Science, Technology, and Communications    | 1\.0                | 1\.0             | 1\.0         | 1               |
| 18             | Foreign Trade                                     | 1\.0                | 0\.25            | 0\.4         | 4               |
| 19             | International Affairs                             | 0\.77               | 0\.71            | 0\.74        | 14              |
| 20             | Government Operations                             | 0\.74               | 0\.45            | 0\.56        | 31              |
| 21             | Public Lands and Water Management                 | 0\.25               | 1\.0             | 0\.4         | 1               |
| 23             | Cultural Policy Issues                            | 1\.0                | 0\.25            | 0\.4         | 4               |

