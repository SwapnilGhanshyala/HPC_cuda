DataRaces:
1) conditions hold simultaneously:
    i) multiple threads
    ii) common memory location
    iii) Atleast 1 write
    iv) Concurrent execution : simutaneously execution of those multiple threads.
2) Solutions:
    i) Avoidance : avoid atleast 1 of the 4 conditions above.
    ii)  Execute sequentiallu
    iii) Privatization / Data Repetions: keep local copy of the shared data
    iv) Separation of Reads and Writes by a barrier, do all reads together and all writes separately 
    v) Mutual exclusion : atomic read write
3) volatile variables, tell the compiler to make sure that it is always reading the latest version of the variable from the global memory.

TODO:
1) Conditions for a Deadlock
A deadlock can occur if these four conditions hold simultaneously:
    i) Mutual Exclusion: Resources involved are non-shareable.
    ii) Hold and Wait: Processes holding resources can request new resources.
    iii) No Preemption: Resources cannot be forcibly taken from processes holding them.
    iv) Circular Wait: A circular chain of processes exists where each process holds a resource needed by the next process.