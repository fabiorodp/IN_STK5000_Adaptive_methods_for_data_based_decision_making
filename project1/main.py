try:
    from task1a import task1a
    from task1b import task1b
    from task1c import task1c
    from task2 import task2
except:
    from project1.task1a import task1a
    from project1.task1b import task1b
    from project1.task1c import task1c
    from project1.task2 import task2

res = input('Which task do you want to perform? (1a, 1b, 1c, 2)')

if res == '1a':
    task1a()

elif res == '1b':
    task1b()

elif res == '1c':
    task1c()

elif res == '2':
    task2()

else:
    print('ERROR: Given task not found!')
