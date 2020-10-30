# Turkiye Student Evaluation Analysis - Clustering

**Complete Video Tutorial:** https://youtu.be/aLEy3GFGWjQ

# Dataset Information

   This data set contains a total 5820 evaluation scores provided by students from Gazi University in Ankara (Turkey). There is a total of 28 course specific questions and additional 5 attributes.

#### Attribute Information:

instr: Instructor's identifier; values taken from {1,2,3} \
class: Course code (descriptor); values taken from {1-13} \
repeat: Number of times the student is taking this course; values taken from {0,1,2,3,...}\
attendance: Code of the level of attendance; values from {0, 1, 2, 3, 4}\
difficulty: Level of difficulty of the course as perceived by the student; values taken from {1,2,3,4,5}\
Q1: The semester course content, teaching method and evaluation system were provided at the start.\
Q2: The course aims and objectives were clearly stated at the beginning of the period.\
Q3: The course was worth the amount of credit assigned to it.\
Q4: The course was taught according to the syllabus announced on the first day of class.\
Q5: The class discussions, homework assignments, applications and studies were satisfactory.\
Q6: The textbook and other courses resources were sufficient and up to date.\
Q7: The course allowed field work, applications, laboratory, discussion and other studies.\
Q8: The quizzes, assignments, projects and exams contributed to helping the learning.\
Q9: I greatly enjoyed the class and was eager to actively participate during the lectures.\
Q10: My initial expectations about the course were met at the end of the period or year.\
Q11: The course was relevant and beneficial to my professional development.\
Q12: The course helped me look at life and the world with a new perspective.\
Q13: The Instructor's knowledge was relevant and up to date.\
Q14: The Instructor came prepared for classes.\
Q15: The Instructor taught in accordance with the announced lesson plan.\
Q16: The Instructor was committed to the course and was understandable.\
Q17: The Instructor arrived on time for classes.\
Q18: The Instructor has a smooth and easy to follow delivery/speech.\
Q19: The Instructor made effective use of class hours.\
Q20: The Instructor explained the course and was eager to be helpful to students.\
Q21: The Instructor demonstrated a positive approach to students.\
Q22: The Instructor was open and respectful of the views of students about the course.\
Q23: The Instructor encouraged participation in the course.\
Q24: The Instructor gave relevant homework assignments/projects, and helped/guided students.\
Q25: The Instructor responded to questions about the course inside and outside of the course.\
Q26: The Instructor's evaluation system (midterm and final questions, projects, assignments, etc.) effectively measured the course objectives.\
Q27: The Instructor provided solutions to exams and discussed them with students.\
Q28: The Instructor treated all students in a right and objective manner.

Q1-Q28 are all Likert-type, meaning that the values are taken from {1,2,3,4,5}

**Download link:** http://archive.ics.uci.edu/ml/machine-learning-databases/00262/

# Libraries

<li>pandas
<li>matplotlib
<li>seaborn
<li>scikit-learn
<li>scipy

# Algorithms

<li>Principal Component Analysis
<li>Kmeans Clustering
<li>Agglomerative Clustering