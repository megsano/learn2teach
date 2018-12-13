# learn2teach
An AI agent that fosters game learning by giving advice and hints at the right times to maximize students' learning.
(The official Stockfish chess engine is required.) 

### Parameter tuning
- To tune either the student DQN or the teacher DQN's hyperparameters, using SGD, run tuning.py 

### Training
- To train the student DQN in Condition 1 or Condition 3, run student_train.py with the appropriate flag: (set with_teacher to True for Condition 1) 
- To train the student DQN in Condition 2, run random_student_train.py 
- To train the teacher DQN, run teacher_train.py
- To test a student DQN in any condition, run test.py (make sure to load the right weight files)
