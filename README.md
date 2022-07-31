# ML-WAF-against-injection-attacks
# ML-WAF-against-injection-attacks
In this project I'm trying to create machine learning WAF against injection attacks:

1. Classical SQL-injections (in-band): error-based and union-based.

2. Blind SQL-injections (blind): time-based and boolean-based.

3. Obfuscated SQL-injections (blacklist).

4. Cross-site scripting (XSS).

5. Directory traversal attack (directory traversal).

6. Server-side template injection (SSTI).

7. Benign traffic (benign).

There are used some metrics:
1. Balanced Accuracy
2. Micro Precision
3. Micro Recall
4. Micro F1-score
5. Macro Precision
6. Macro Recall
7. Macro F1-score
8. Weighted Precision
9. Weighted Recall
10. Weighted F1-score
11. MCC (Matthews Correlation Coefficient)
12. Youden's J statistic

And some graphics:
1. Micro and Macro ROC-curves. Macro ROC-curve was made using linear interpolation.
2. ROC-curves for every class with it's best threshold
3. Commented out graphic for cheking if model is overfitting or underfitting (underfit_or_overfit func)
4. Precision-recall curves with it's best threshold

Examples for creating datasets:

SSTI - https://github.com/swisskyrepo/PayloadsAllTheThings/tree/master/Server%20Side%20Template%20Injection

XSS - https://portswigger.net/web-security/cross-site-scripting/cheat-sheet

Benign traffic - https://www.kaggle.com/code/syedsaqlainhussain/sql-injection-dectection-using-neural-network/notebook

Blacklist - https://owasp.org/www-community/attacks/SQL_Injection_Bypassing_WAF

SQL-injections (in-band) - https://www.kaggle.com/code/syedsaqlainhussain/sql-injection-dectection-using-neural-network/data

SQL-injections (blind) - https://www.kaggle.com/datasets/sajid576/sql-injection-dataset?select=Modified_SQL_Dataset.csv

Directory traversal - https://github.com/omurugur/Path_Travelsal_Payload_List/blob/master/Payload/Deep-Travelsal.txt

Video-demo - https://youtu.be/mOANuc6V80U 
