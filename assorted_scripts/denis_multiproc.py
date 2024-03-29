import multiprocessing as mp
import tensorflow as tf

#=========1=========2=========3=========4=========5=========6=========7=

def _threadedFunc(x, out_q):
   name = mp.current_process().name
   print(name, 'Starting')
   sess = tf.Session()
   result = sess.run(tf.add(x, 3))
   out_q.put(result)
   print(name, 'Exiting')

q = mp.Queue()
t1 = mp.Process(target=_threadedFunc, args=(1, q))
t2 = mp.Process(target=_threadedFunc, args=(2, q))
t1.start()
t2.start()
t2.join()
t1.join()

# will print 4 and 5 in some order
print(q.get())
print(q.get())
