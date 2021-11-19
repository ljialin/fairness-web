from multiprocessing import Process,Queue
import time
import random

def fun1(q,i):
    print('子进程%s 开始put数据' %i)
    for a in range(2):
        q.put('我是{} 通过Queue进行第{}次通信'.format(i,a))
        time.sleep(2 + random.random())
    q.put("进程{}结束".format(i))

if __name__ == '__main__':
    q = Queue()

    process_list = []
    for i in range(3):
        p = Process(target=fun1,args=(q,i,))  #注意args里面要把q对象传给我们要执行的方法，这样子进程才能和主进程用Queue来通信
        p.start()
        process_list.append(p)


    print('主进程获取Queue数据')
    count = 0
    while 1:
        text = q.get(timeout=10086)
        print(text)
        if "结束" in text:
            count += 1
        if count > 2: break
    print('结束全部测试')

    for i in process_list:
        p.join()