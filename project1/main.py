try:
    from task1a import task1a
    from task1b import task1b
    from task1c import task1c
    from task2a import task2a
    from task2b import task2b
except:
    from project1.task1a import task1a
    from project1.task1b import task1b
    from project1.task1c import task1c
    from project1.task2a import task2a
    from project1.task2b import task2b

res = input('Which task do you want to perform? (1a, 1b, 1c, 2a, 2b)')

if res == '1a':
    task1a()

elif res == '1b':
    task1b()

elif res == '1c':
    task1c()

elif res == '2a':
    task2a()

elif res == '2b':
    task2b()

else:
    print('ERROR: Given task not found!')
