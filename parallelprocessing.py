import multiprocessing

def put_func_to_queue(func,job_name,queue):
    ''' A decorator for submitting job to queue
    '''
    def new_func(*args,**kwargs):
        try:
            queue.put((job_name,func(*args,**kwargs)))
        except Exception as e:
            print func.__name__ +":"+str(e)
            queue.put((job_name,e))
        return None
    return new_func


def run_in_parallel(func):
    ''' Decorator for running func in parallel
    Each time this decorator is called a new multiprocessing.Queue is created
    And the queue is to be closed by extract_output(ps,queue_output)
    Return: function(**kwargs) which returns (ps,queue_outout)
    ps is a list of Process and queue_output is a multiprocessing.Queue
    '''
    queue_output = multiprocessing.Queue()
    ps = []
    def run_func(*args,**kwargs):
        ps.append(multiprocessing.Process(target=put_func_to_queue(func,len(ps),
                                                                   queue_output),
                                          args=args,kwargs=kwargs))
        ps[-1].start()
        return ps,queue_output
    return run_func


def extract_output(ps,queue_output,timeout=None):
    ''' Extract the output from run_in_parallel
    in the order when the job is declared and submitted
    Return : list
    '''
    results = []
    for p in ps:
        results.append(queue_output.get(timeout=timeout))
    # empty all processes
    while ps:
        ps.pop().join()
    try:
        indices,results =  zip(*sorted(results,key=lambda a: a[0]))
    except ValueError:
        # No result is returned
        pass
    return list(results)

