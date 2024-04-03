def fibonacci(n):
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a + b
    print()

fibonacci(100)  # Пример вывода чисел Фибоначчи до 100
