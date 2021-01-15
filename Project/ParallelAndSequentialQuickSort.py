import random, time, sys
from multiprocessing import Process, Pipe
import matplotlib.pyplot as plt

# Start of the parallel and sequential quick sorts
def main():
    """
    This is the main method, where we:
    -generate a random list.
    -time a sequential quicksort on the list.
    -time a parallel quicksort on the list.
    -time Python's built-in sorted on the list.
    """
    N = 5000000
    if len(sys.argv) > 1:  # the user input a list size.
        N = int(sys.argv[1])

    # We want to sort the same list, so make a backup.
    lystbck = [random.randint(0, N) for x in range(N)]
    # print("\nOur unsorted input list: \n" + str(lystbck))

    lyst = list(lystbck)
    graphParallelQuickSort(lyst)


    #
    # Sequential QuickSort
    #

    # Sequential quicksort a copy of the list.
    # lyst = list(lystbck)  # copy the list
    # start = time.time()  # start time
    # print("\nWe are starting sequential Quick Sort...")
    # lyst = quicksort(lyst)  # quicksort the list
    # print("Finished sequential Quick Sort!")
    # elapsed = time.time() - start  # stop time
    # if not isSorted(lyst):
    #     print('\nquicksort did not sort the lyst. oops.\n')
    # print('Sequential Quick Sort time: %f sec' % (elapsed))
    # #print("The sorted list of our input list using sequential quick sort: \n" + str(lyst))

    #
    # Parallel QuickSort
    #

    # So that cpu usage shows a lull.
    # time.sleep(3)

    # Parallel quicksort.
    # lyst = list(lystbck)
    # start = time.time()
    # print("\nStarting parallel Quick Sort...")
    # # 2**(n+1) - 1 processes will be instantiated.
    # # I set the number of processes to be high since, with
    # # a random choice of pivot, it is unlikely the work
    # # will distribute evenly.
    # n = 8
    # # Instantiate a Pipe so that we can receive the
    # # process's response.
    # pconn, cconn = Pipe()
    # # Instantiate a process that executes quicksort Parallel
    # # on the entire list.
    # p = Process(target=quicksortParallel, \
    #             args=(lyst, cconn, n))
    # p.start()
    # lyst = pconn.recv()
    # # Blocks until there is something (the sorted list)
    # # to receive.
    # p.join()
    # elapsed = time.time() - start
    # print("Finished parallel Quick Sort!")
    #
    # if not isSorted(lyst):
    #     print('\nParallel Quick Sort did not sort the lyst. oops.\n')
    # print('Parallel QuickSort time: %f sec' % (elapsed))
    #
    # #print("The sorted list of our input list using parallel quick sort: \n" + str(lyst))
    # time.sleep(3)

    #
    # Built-in Python sorting function
    #

    # # Built-in test.
    # # The underlying c code is obviously the fastest, but then
    # # using a calculator is usually faster too.  That isn't the
    # # point here obviously.
    # lyst = list(lystbck)
    # start = time.time()
    # print("\nStarting the built in sort...")
    # lyst = sorted(lyst)
    # elapsed = time.time() - start
    # print("Finished the built in sort")
    # print('Built-in sort time: %f sec' % (elapsed))
    # #print("The sorted list of our input list using built in quick sort: \n" + str(lyst) + "\n")

def graphParallelQuickSort(list):
    timingList = []
    processList = []
    # add to the run time list then we could graph them
    for n in range(0, 9):
        print("Iteration: " + str(n) + "\n")
        start = time.time()
        pconn, cconn = Pipe()
        p = Process(target=quicksortParallel, \
                    args=(list, cconn, n))
        p.start()
        list = pconn.recv()
        p.join()
        elapsed = time.time() - start
        timingList.append(elapsed)
        processList.append((2**(n+1) - 1))


    # Graphing speedup
    plt.figure(figsize=(10,10))
    plt.scatter(processList, timingList)
    plt.plot(processList, timingList)
    plt.xlabel("Number of parallel processes")
    plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 540])
    plt.ylabel("Run time")
    plt.title("Parallel Quick Sort Speedup")
    plt.savefig('Quick_Sort_Speedup.png')
    plt.show()




def quicksort(lyst):
    if len(lyst) <= 1:
        return lyst
    pivot = lyst.pop(random.randint(0, len(lyst) - 1))

    return quicksort([x for x in lyst if x < pivot]) \
           + [pivot] \
           + quicksort([x for x in lyst if x >= pivot])


def quicksortParallel(lyst, conn, procNum):

    """
    Partition the list, then quicksort the left and right
    sides in parallel.
    """

    if procNum <= 0 or len(lyst) <= 1:
        # In the case of len(lyst) <= 1, quicksort will
        # immediately return anyway.
        conn.send(quicksort(lyst))
        conn.close()
        return

    # Create two independent lists (independent in that
    # elements will never need be compared between lists).
    pivot = lyst.pop(random.randint(0, len(lyst) - 1))

    leftSide = [x for x in lyst if x < pivot]
    rightSide = [x for x in lyst if x >= pivot]

    # Creat a Pipe to communicate with the left subprocess
    pconnLeft, cconnLeft = Pipe()
    # Create a leftProc that executes quicksortParallel on
    # the left half-list.
    leftProc = Process(target=quicksortParallel, \
                       args=(leftSide, cconnLeft, procNum - 1))

    # Again, for the right.
    pconnRight, cconnRight = Pipe()
    rightProc = Process(target=quicksortParallel, \
                        args=(rightSide, cconnRight, procNum - 1))

    # Start the two subprocesses.
    leftProc.start()
    rightProc.start()

    # Our answer is the concatenation of the subprocesses'
    # answers, with the pivot in between.
    conn.send(pconnLeft.recv() + [pivot] + pconnRight.recv())
    conn.close()

    # Join our subprocesses.
    leftProc.join()
    rightProc.join()

# Function that returns true if the list is sorted
def isSorted(lyst):
    """
    Return whether the argument lyst is in non-decreasing order.
    """
    for i in range(1, len(lyst)):
        if lyst[i] < lyst[i - 1]:
            return False
    return True

# Call the main method if run from the command line.
if __name__ == '__main__':
    main()


