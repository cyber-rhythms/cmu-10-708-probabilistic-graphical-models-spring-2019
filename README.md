# CMU 10-708 Probabilistic Graphical Models Spring 2019 (Eric Xing)

A fully organised, integrity checked repo containing all the lecture materials available at:

https://sailinglab.github.io/pgm-spring-2019/lectures/

The course videos that accompany the contents of this repo are available here:

https://scs.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx#folderID=%220f44b4d7-fb4e-49eb-b88d-a9d00125e1b3%22&maxResults=50

The video playlist has been cross-checked against the coursepage for completeness. See remarks.

By folder, the repo contains:

1) 10-708-probabilistic-graphical-models-coursepage.html - an archived complete html document
that contains the course schedule, in case the link is no longer hosted by CMU.

2) 10-708-probabilistic-graphical-models-coursepage folder - the relevant html files for the
above complete html document. 

3) homework-assignments folder - contains 4 homework assignment materials*.

4) lecture-slides folder - contains all the lecture slides in PDF form*. File names have been
"cleaned" and associated with a date for ease of association to the main course page.

5) lecture-sldies-annotated folder - contains version of the lecture slides that have been
annotated by the instructor.

6) notes folder - a placeholder folder as a reminder that there exist notes scribed by students
of the course. These are all available as distil style blogs at the following link:

  https://sailinglab.github.io/pgm-spring-2019/notes/

  Each blog/lecture notes page is assembled from the following associated github repo:

  https://github.com/sailinglab/pgm-spring-2019

7) suggested-projects folder - contains an archived complete html webpage of the following link:

  https://sailinglab.github.io/pgm-spring-2019/project/

Remarks:

The reason for archiving course pages and to save the materials is to provide an additional location
for auto-didacts to download the materials, in the event that the sailing.github.io, for whatever
reason, no longer hosts the materials. In the past, some CMU courses have had their pages removed
after the semester has ended.

A copy of all the PDF readings for the course is not uploaded due to repo space constraints, but can
be downloaded by going to the coursepage, and failing that, Googling the titles.

"Lecture #21 Sequential decision marking (part 2): the algorithms" slides were not available from the coursepage as the link was broken.
However, the annotated version of these slides were available, so "lecture21-rl+pgm-pt2-maruan-03-apr-annotated.pdf"
has been placed in "lecture-slides" to reflect this.

"Lecture #5(skipped): Parameter learning in fully observable Bayesian Networks" video is not
in the playlist, as it was skipped, as indicated on the course page. 

Homework assignments - these comprise a mix of theoretical, maths-style derivation questions; and exercises in which a dataset is given and a particular algorithm needs to be implemented in code.

For someone who was enrolled as a CMU student, the derivations would be marked by a teaching
assistant, and the code implementations would be marked by a scripted automatic grading system (Gradescope).

For these to have value to those self-studying, it needs to be ascertained whether it is possible
to do these exercises without these university facilities. Here are my comments from a preliminary audit of the materials.

HW1 - Mix of exercises taken from Koller and Friedman(2009); and implementation of junction tree algorithm and EM
algorithm for parameter estimation of HMM. Script skeletons are provided and student needs to fill in the blanks for submission to Gradescope.

HW2 - Mix of derivations and code implementations of algorithms from selected papers (HMM and CRF for POS tagging;
variational EM for LDA on wikipedia corpus (optional); Metropolis-Hastings algorithm for sports data).
Results are expected as a PDF write-up.

HW3 - Mix of derviations and guided code implementations (Wake-Sleep Algorithm) using provided Jupyter notebooks.
Code is used to create results that are to be written up. Results are expected as a submitted Jupyter/Colab notebook.

HW4 - Mixture of theoretical derivations and empirical questions. Pre-written code is provided in a Colab/Jupyter Notebook. Expected
to run tests, interpret results. No coding implementation questions that need to be submitted to an autograder.

Homework assignment audit conclusion: without having done the exercises, it does seem firmly
within the realm of possibility for a motivated student to undertake these exercises and derive value
from them.

For derivation questions - these generally tend to be doable, and success here depends on how well one has assimilated the contents of the lectures. 

For code/implementations of algorithms that would normally be marked by Gradescope - there are often mathematical constraints on the behaviour of the output of the algorithm that allow one to check if it is working correctly. The papers on which the exercises are based will contain a lot of supporting information to assist also. A source of "ground-truth" might be using a complete implementation of an algorithm in an external package ONLY as a checking mechanism. One could then take test input data, run our hand-coded algorithm on that data; and compare the results, against the external package implementation, provided that there are sufficient grounds to enable comparison.

For empirical write-ups - there are often a lot of the clues in the context of the problem that may be helpful.

Ultimately, the student doing these exercises without university support will indeed be working "blind", but the greater takeaway is that if the student is successful, the skills picked up will be immensely beneficial. Real-world problems have no instructor, nor automated systems telling the student whether they have indeed "scored the correct answer". That places the onus on the student to go and assess whether they have indeed gotten the right answer, and is an arguably greater test of resourcefulness.
